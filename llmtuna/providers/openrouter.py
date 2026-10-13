"""OpenRouter provider — talks to OpenRouter via the OpenAI Python SDK.

OpenRouter is the OpenAI-compatible aggregator recommended for v1. The
class translates llmtuna's neutral tool spec into OpenAI's function-
calling format, forces the LLM to call the proposal tool, captures any
reasoning tokens the model emits, and returns the parsed result in the
``Provider.propose`` shape.

Vendor-specific exceptions from the openai SDK (``openai.BadRequestError``
for context overflow, ``openai.APIError`` for network/server issues,
etc.) propagate unchanged. Catch them at the call site if you need to
handle specific failure modes.
"""

import json
import os

from openai import OpenAI

from llmtuna.providers.base import Provider


class OpenRouter(Provider):
    """Provider that calls OpenRouter via the OpenAI Python SDK.

    Forces the LLM to call the provided tool via ``tool_choice`` and
    captures reasoning tokens when the model supports them.

    Attributes:
        model: OpenRouter model id, e.g. ``"anthropic/claude-sonnet-4-6"``.
        thinking_budget: Max reasoning tokens. ``0`` disables reasoning.
        max_tokens: Max output tokens for the model's response.
        max_retries: Retries on empty/no-choice responses (transient).
        extra_body: Dict forwarded as raw JSON in the request body for
            vendor-specific knobs.
        base_url: API endpoint (defaults to OpenRouter).
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        thinking_budget: int = 10000,
        max_tokens: int = 2000,
        max_retries: int = 3,
        force_tool: bool = False,
        extra_body: dict | None = None,
        base_url: str | None = None,
    ):
        """Initialize the OpenRouter provider.

        Args:
            model: OpenRouter model id (e.g. ``"anthropic/claude-sonnet-4-6"``).
            api_key: API key. Falls back to ``OPENROUTER_API_KEY`` env var.
            thinking_budget: Max reasoning tokens. Set ``0`` to disable
                reasoning (faster, cheaper, no thinking trace).
            max_tokens: Max output tokens.
            max_retries: Number of times to retry on transient empty
                responses (API returned no choices).
            force_tool: If True, send OpenAI-style ``tool_choice`` to
                force the LLM to call ``propose_config``. **Cannot be
                combined with extended thinking** (Anthropic's API, for
                one, rejects forced tool use when thinking is on), so it
                requires ``thinking_budget=0``. Default ``False`` — relies
                on the system-prompt instruction to call the tool, which
                strong models honor reliably and which preserves the
                model's reasoning trace. **You can have forced tool
                calls OR captured reasoning, not both.**
            extra_body: Dict forwarded as raw JSON in the request body
                for vendor-specific knobs. The reasoning config is
                injected automatically when ``thinking_budget > 0``;
                user-supplied keys take precedence.
            base_url: API endpoint. Defaults to OpenRouter; set to
                another OpenAI-compatible endpoint to redirect.

        Raises:
            ValueError: If no API key is found (neither ``api_key`` nor
                ``OPENROUTER_API_KEY`` env var); or if both
                ``force_tool=True`` and ``thinking_budget > 0`` are set,
                since forced tool calls and extended thinking cannot be
                combined.
        """
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter: no API key — pass api_key= or set OPENROUTER_API_KEY"
            )
        if force_tool and thinking_budget > 0:
            raise ValueError(
                "OpenRouter: force_tool=True cannot be combined with extended "
                "thinking. Set thinking_budget=0 if you want forced tool calls, "
                "or force_tool=False (default) to keep reasoning."
            )
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.force_tool = force_tool
        self.extra_body = extra_body or {}
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._client = OpenAI(base_url=self.base_url, api_key=key)

    def propose(self, system: str, user: str, tool_spec: dict) -> dict:
        """Call the LLM and return the parsed proposal.

        Sends the tool spec, optionally with ``tool_choice`` forcing
        (controlled by ``force_tool``), retries up to ``max_retries``
        times on empty responses, then parses the message into the
        ``Provider.propose`` shape.

        If the LLM did not call the tool, returns an empty ``tool_args``
        dict — the Tuner's validation retry then re-prompts naturally
        rather than crashing the loop.

        Args:
            system: System prompt.
            user: User message.
            tool_spec: Neutral tool spec
                ``{"name": str, "description": str, "parameters": <JSON schema>}``.
                Translated to OpenAI's function-calling format internally.

        Returns:
            ``{"reasoning": str, "content": str, "tool_args": dict}``.
            ``tool_args`` is ``{}`` if the model produced no tool call.

        Raises:
            ValueError: If tool arguments are not valid JSON (retried by
                ``Tuner.suggest``).
            RuntimeError: If the API returned no choices after
                ``max_retries`` attempts.
            openai.OpenAIError: Vendor SDK exceptions (network errors,
                bad-request, context overflow, etc.) propagate unchanged.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        oai_tool = {
            "type": "function",
            "function": {
                "name": tool_spec["name"],
                "description": tool_spec["description"],
                "parameters": tool_spec["parameters"],
            },
        }
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "tools": [oai_tool],
        }
        if self.force_tool:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_spec["name"]},
            }
        body = dict(self.extra_body)
        if self.thinking_budget > 0:
            body.setdefault("reasoning", {"max_tokens": self.thinking_budget})
        if body:
            kwargs["extra_body"] = body

        response = None
        for _ in range(self.max_retries):
            response = self._client.chat.completions.create(**kwargs)
            if response.choices:
                break
        if response is None or not response.choices:
            raise RuntimeError(
                f"OpenRouter: no choices returned after {self.max_retries} attempts"
            )

        return self._parse_message(response.choices[0].message)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Free-form text generation (no tool call). See ``Provider.complete``.

        Used for summarization and other one-shot text tasks. Reasoning
        is intentionally disabled here — these calls are usually quick
        digests where thinking overhead isn't worth the cost.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    def _parse_message(self, msg) -> dict:
        """Parse an OpenAI-compatible message into the ``Provider.propose`` shape.

        Returns ``tool_args = {}`` when the LLM produced no tool call,
        rather than raising — the Tuner's validation retry then
        re-prompts naturally rather than crashing the loop.

        Args:
            msg: The ``message`` field from a chat completions response.

        Returns:
            ``{"reasoning": str, "content": str, "tool_args": dict}``.
            ``tool_args`` is ``{}`` if no tool call was emitted.

        Raises:
            ValueError: If a tool call is present but its arguments
                are not valid JSON.
        """
        reasoning = ""
        if getattr(msg, "reasoning", None):
            reasoning = msg.reasoning
        elif getattr(msg, "reasoning_details", None):
            for detail in msg.reasoning_details:
                if hasattr(detail, "text"):
                    reasoning += detail.text or ""
                elif isinstance(detail, dict):
                    kind = detail.get("type")
                    if kind == "reasoning.text":
                        reasoning += detail.get("text", "")
                    elif kind == "reasoning.encrypted":
                        reasoning += "[reasoning encrypted by provider]"

        content = msg.content or ""

        if not getattr(msg, "tool_calls", None):
            return {"reasoning": reasoning, "content": content, "tool_args": {}}

        tc = msg.tool_calls[0]
        try:
            tool_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            # ValueError, so Tuner.suggest retries it like any malformed proposal.
            raise ValueError(
                "OpenRouter: tool arguments not valid JSON: "
                f"{tc.function.arguments[:200]!r}"
            ) from e

        return {
            "reasoning": reasoning,
            "content": content,
            "tool_args": tool_args,
        }
