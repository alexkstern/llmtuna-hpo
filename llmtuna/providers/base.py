"""Provider — vendor-neutral abstract base class for LLM backends.

A Provider takes a system prompt, a user message, and a tool spec, forwards
them to an LLM backend, and returns the parsed arguments of the LLM's
forced tool call as a plain dict mapping hparam name to value.

All vendor-specific concerns — API client setup, tool-format translation,
response parsing, reasoning-token configuration, retry-on-malformed-output —
live inside concrete subclasses (e.g. ``OpenRouter``, ``MockProvider``).
"""

from abc import ABC, abstractmethod


class Provider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement ``propose()``. The contract is intentionally
    narrow: take a tool definition, return parsed tool arguments as a dict.
    Vendor-specific shape translation lives inside the subclass.
    """

    @abstractmethod
    def propose(
        self,
        system: str,
        user: str,
        tool_spec: dict,
    ) -> dict:
        """Forward the call to the LLM and return its tool call arguments.

        Args:
            system: System prompt sent to the LLM.
            user: User message (already-rendered context plus instructions).
            tool_spec: Tool definition the LLM is forced to call. Shape:
                ``{"name": str, "description": str, "parameters": <JSON schema>}``.
                Subclasses translate this to the vendor's tool format.

        Returns:
            A dict with three keys:

            - ``"reasoning"`` (str): The LLM's reasoning chain. Empty
              string if reasoning was disabled or unsupported by the model.
            - ``"content"`` (str): Any non-tool text the LLM emitted.
              Often empty when the tool call is forced.
            - ``"tool_args"`` (dict): The parsed tool call arguments
              mapping hparam name to value, e.g.
              ``{"lr": 1e-3, "depth": 12}``.

            All three keys are always present. The Tuner appends the full
            response (including reasoning) to the Context so the LLM
            sees its own prior thinking on subsequent calls.

        Raises:
            Exception: Subclasses signal failures by raising — the type
                varies by backend (network errors, malformed responses,
                test misconfiguration). See each concrete provider's
                docstring for specifics.
        """

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Run a single text-generation call (no tool, no structured output).

        Used by ``Context.add_summary`` and any other helper that wants
        plain text from the LLM rather than a forced tool call.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Optional cap on output tokens. When ``None``, the
                provider's own default is used.

        Returns:
            The model's text content as a string. May be empty if the
            model returned nothing.
        """
