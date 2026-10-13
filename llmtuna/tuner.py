"""Tuner — the main optimizer class.

A ``Tuner`` ties together a ``Provider``, a list of ``Param`` definitions
(the search space), and a ``Context`` (the rolling text log shown to the
LLM). The user calls ``suggest()`` to get a configuration to try, runs
their training, and reports the result via ``observe()``.

The ``best`` property and ``history`` list expose structured results;
``save()``/``load()`` serialize the full state to JSON for resume or
audit.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from llmtuna import defaults
from llmtuna.context import Context
from llmtuna.providers.base import Provider
from llmtuna.space import Param, param_from_dict, param_to_dict


@dataclass
class Trial:
    """One observed trial — a configuration and the metric it produced.

    Attributes:
        cfg: The hyperparameter configuration that was tried.
        value: The empirical metric the user reported.
        note: Optional free-form note (e.g. ``"diverged at step 1200"``).
    """

    cfg: dict
    value: float
    note: str | None = None


class Tuner:
    """LLM-driven hyperparameter optimizer.

    Workflow:

    1. Construct a ``Tuner`` with a ``Provider``, a list of ``Param``
       definitions, and an objective.
    2. Optionally seed ``self.context`` with architecture notes, prior
       results, paper findings, etc. via ``add()`` / ``add_file()``.
    3. In a loop: ``cfg = opt.suggest()``, run training, then
       ``opt.observe(cfg, value, note=...)``.
    4. Inspect ``opt.best`` / ``opt.history`` at any time, or ``save()``
       to disk for resume or audit.

    Attributes:
        provider: The ``Provider`` used to call the LLM.
        space: The list of ``Param`` definitions defining the search space.
        objective: Either ``"minimize"`` or ``"maximize"``.
        max_retries: Number of retries on validation failure per
            ``suggest()`` call.
        context: The rolling ``Context`` shown to the LLM.
        history: List of observed ``Trial`` records.
        system_prompt: System prompt sent to the LLM (string).
        format_proposal: Callable(response_dict) -> str — formats raw LLM
            response as a context entry.
        format_result: Callable(cfg, value, note) -> str — formats trial
            result as a context entry.
        build_user_message: Callable(context_text, space_summary,
            objective) -> str — assembles the user message.
    """

    def __init__(
        self,
        provider: Provider,
        space: list[Param],
        *,
        objective: Literal["minimize", "maximize"] = "minimize",
        max_retries: int = 3,
        system_prompt: str | None = None,
        format_proposal: Callable[[dict], str] | None = None,
        format_result: Callable[[dict, float, str | None], str] | None = None,
        build_user_message: Callable[[str, list[str], str], str] | None = None,
    ):
        """Initialize a new Tuner.

        Args:
            provider: A concrete ``Provider`` (e.g. ``OpenRouter``,
                ``MockProvider``).
            space: List of ``Param`` definitions (``Float``, ``Int``,
                ``Choice``). Must be non-empty; names must be unique.
            objective: ``"minimize"`` or ``"maximize"``. Determines how
                ``best`` selects the winning trial.
            max_retries: Max number of retries on validation failure
                during ``suggest()``. Each retry feeds the validation
                error back to the LLM as a follow-up message.
            system_prompt: Override for the default system prompt.
            format_proposal: Override for the default proposal formatter.
            format_result: Override for the default result formatter.
            build_user_message: Override for the default user-message
                builder.

        Raises:
            ValueError: If ``space`` is empty, has duplicate names, or
                ``objective`` is not ``"minimize"`` / ``"maximize"``.
        """
        if not space:
            raise ValueError("Tuner: space must contain at least one parameter")
        names = [p.name for p in space]
        if len(set(names)) != len(names):
            raise ValueError(f"Tuner: duplicate parameter names in space: {names}")
        if objective not in ("minimize", "maximize"):
            raise ValueError(
                f"Tuner: objective must be 'minimize' or 'maximize', got {objective!r}"
            )

        self.provider = provider
        self.space = space
        self.objective = objective
        self.max_retries = max_retries
        self.system_prompt = system_prompt or defaults.SYSTEM_PROMPT
        self.format_proposal = format_proposal or defaults.format_proposal
        self.format_result = format_result or defaults.format_result
        self.build_user_message = build_user_message or defaults.build_user_message
        self.context = Context()
        self.history: list[Trial] = []

    def _tool_spec(self) -> dict:
        """Build the tool definition the LLM is forced to call."""
        return {
            "name": "propose_config",
            "description": (
                "Propose the next hyperparameter configuration. "
                "Include a value for every parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {p.name: p.to_schema() for p in self.space},
                "required": [p.name for p in self.space],
            },
        }

    def _validate_tool_args(self, tool_args: dict) -> dict:
        """Validate every space param against the LLM's tool_args.

        Args:
            tool_args: The dict returned by the LLM's tool call.

        Returns:
            A new dict mapping each param name to its validated/coerced value.

        Raises:
            ValueError: On the first validation failure encountered. Caught
                by ``suggest()``'s retry loop and fed back to the LLM as a
                correction message.
        """
        if not isinstance(tool_args, dict):
            raise ValueError(
                f"tool_args must be a JSON object, got {type(tool_args).__name__}."
            )
        expected = {p.name for p in self.space}
        extras = set(tool_args) - expected
        if extras:
            raise ValueError(
                f"Unexpected parameters in tool_args: {sorted(extras)}. "
                f"The search space defines only: {sorted(expected)}."
            )
        cfg: dict[str, Any] = {}
        for p in self.space:
            if p.name not in tool_args:
                raise ValueError(
                    f"Missing required parameter '{p.name}' in tool_args"
                )
            cfg[p.name] = p.validate(tool_args[p.name])
        return cfg

    def suggest(self) -> dict:
        """Ask the LLM to propose the next hyperparameter configuration.

        Builds a user message from the current context plus search-space
        summary, calls the provider, validates the returned values against
        the space, and on validation failure retries with the error fed
        back to the LLM. The full provider response (reasoning + content +
        tool_args) is appended to the context on every attempt.

        Returns:
            A dict mapping each parameter name to its validated value.

        Raises:
            RuntimeError: If validation fails on every attempt
                (initial + ``max_retries``).
        """
        base_user_msg = self.build_user_message(
            context_text=self.context.render(),
            space_summary=[p.summary() for p in self.space],
            objective=self.objective,
        )
        tool_spec = self._tool_spec()

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            user_msg = base_user_msg
            if last_error is not None:
                user_msg = (
                    base_user_msg
                    + f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease retry."
                )

            try:
                response = self.provider.propose(
                    system=self.system_prompt,
                    user=user_msg,
                    tool_spec=tool_spec,
                )
            except ValueError as e:  # malformed provider output, e.g. invalid tool JSON
                last_error = str(e)
                continue

            self.context.add(text=self.format_proposal(response))

            try:
                return self._validate_tool_args(tool_args=response["tool_args"])
            except ValueError as e:
                last_error = str(e)

        raise RuntimeError(
            f"Tuner.suggest: validation failed on all "
            f"{self.max_retries + 1} attempts. Last error: {last_error}"
        )

    def render_prompt(self) -> dict:
        """Return the full prompt that ``suggest()`` would send, without sending it.

        Builds the system prompt, user message, and tool spec exactly as
        ``suggest()`` would assemble them at this moment, and returns
        them as a JSON-safe dict. **No provider call is made** — useful
        for previewing what the LLM is about to see, debugging, or
        snapshotting for audit via ``save_prompt()``.

        Returns:
            A dict with three keys, all JSON-serializable:

            - ``"system"`` (str): The system prompt.
            - ``"user"`` (str): The fully assembled user message
              (parameter bounds prelude + rendered context + closing
              instruction).
            - ``"tool_spec"`` (dict): The ``propose_config`` tool
              definition (name, description, JSON-Schema parameters).
        """
        return {
            "system": self.system_prompt,
            "user": self.build_user_message(
                context_text=self.context.render(),
                space_summary=[p.summary() for p in self.space],
                objective=self.objective,
            ),
            "tool_spec": self._tool_spec(),
        }

    def save_prompt(self, path: str | Path) -> None:
        """Write the current ``render_prompt()`` output to a JSON file.

        Sugar over ``json.dump(self.render_prompt(), open(path, "w"))``.
        The resulting file is a frozen snapshot of what the LLM would
        see on the next ``suggest()`` — for human inspection / audit.
        It is **not** loadable; use ``save()`` / ``load()`` for state
        round-trip.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.render_prompt(), indent=2))

    def observe(
        self,
        cfg: dict,
        value: float,
        note: str | None = None,
    ) -> None:
        """Record the result of a trial.

        Appends a ``Trial`` to ``history`` and a formatted result entry
        to ``context`` (so the LLM sees it on the next ``suggest()``).

        Args:
            cfg: The configuration that was tried (typically from a prior
                ``suggest()``).
            value: The empirical metric the user reported.
            note: Optional free-form note to include with this trial.
        """
        self.history.append(Trial(cfg=cfg, value=value, note=note))
        self.context.add(
            text=self.format_result(cfg=cfg, value=value, note=note)
        )

    @property
    def best(self) -> dict | None:
        """The best observed trial so far, sign-aware on ``objective``.

        Returns:
            A dict ``{"cfg": ..., "value": ...}`` for the winning trial,
            or ``None`` if no trials have been observed yet. On ties,
            returns the first matching trial in chronological order.
        """
        if not self.history:
            return None
        winner = (min if self.objective == "minimize" else max)(
            self.history, key=lambda t: t.value
        )
        return {"cfg": winner.cfg, "value": winner.value}

    def save(self, path: str | Path) -> None:
        """Serialize Tuner state to a JSON file.

        The provider is intentionally NOT serialized (no API keys on
        disk). On reload, the user supplies a fresh provider via
        ``Tuner.load(path, provider=...)``.

        Args:
            path: Destination file path.
        """
        data = {
            "version": 1,
            "objective": self.objective,
            "max_retries": self.max_retries,
            "system_prompt": self.system_prompt,
            "space": [param_to_dict(p) for p in self.space],
            "context": self.context.to_dict(),
            "history": [asdict(t) for t in self.history],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        provider: Provider,
        *,
        system_prompt: str | None = None,
        format_proposal: Callable[[dict], str] | None = None,
        format_result: Callable[[dict, float, str | None], str] | None = None,
        build_user_message: Callable[[str, list[str], str], str] | None = None,
    ) -> "Tuner":
        """Reconstruct a Tuner from a JSON file written by ``save()``.

        ``system_prompt`` IS serialized and restored
        automatically. Pass ``system_prompt=`` here only if you want to
        OVERRIDE the saved value. Custom formatters are NOT serialized
        (functions can't round-trip through JSON); if the original
        Tuner used custom formatters, **pass them again here** —
        otherwise the defaults from ``llmtuna.defaults`` are used and
        you will silently lose your customizations.

        Args:
            path: Source file path.
            provider: A fresh ``Provider`` to attach. The provider is NOT
                serialized in the save file (no API keys on disk).
            system_prompt: Override for the saved system prompt. If
                ``None`` and the save file contains one, the saved
                value is used. If ``None`` and the save file lacks
                one (older saves), ``defaults.SYSTEM_PROMPT`` is used.
            format_proposal: Override for the default proposal formatter.
            format_result: Override for the default result formatter.
            build_user_message: Override for the default user-message builder.

        Returns:
            A new ``Tuner`` with the saved space, objective, context,
            history, and system prompt, plus any overrides you supplied.
        """
        data = json.loads(Path(path).read_text())
        space = [param_from_dict(d) for d in data["space"]]
        opt = cls(
            provider=provider,
            space=space,
            objective=data["objective"],
            max_retries=data["max_retries"],
            system_prompt=(
                system_prompt
                if system_prompt is not None
                else data.get("system_prompt")
            ),
            format_proposal=format_proposal,
            format_result=format_result,
            build_user_message=build_user_message,
        )
        opt.context = Context.from_dict(data=data["context"])
        opt.history = [Trial(**t) for t in data["history"]]
        return opt
