"""Default system prompt and formatter functions used by ``Tuner``.

Each value/function here is overridable via ``Tuner`` kwargs
(``system_prompt``, ``format_proposal``, ``format_result``,
``build_user_message``). They are exposed as plain top-level constants and
functions so they can be swapped, unit-tested, or compiled by future
prompt-optimization tools (DSPy, TextGrad, etc.) without touching the
``Tuner`` itself.
"""

SUMMARIZE_SYSTEM = """\
You are a senior ML engineer. You will be given source code files from a \
training or experiment codebase, along with the names of the \
hyperparameters that are being optimized over.

Produce a CONCISE summary (target 300-500 words) covering ONLY what is \
relevant to choosing good values for those hyperparameters:

1. **What the code does**: model / system architecture, data, what is \
   being trained or evaluated, and what metric is reported.
2. **Training / evaluation setup**: optimizer, schedules, key \
   computational details that affect hparam choice.
3. **For each tuned hparam**: what it controls in this code, sensible \
   ranges if obvious from the code, and any couplings with other \
   tuned hparams.
4. **Loss landscape intuition**: plausible interactions, log-scale vs \
   linear sensitivity, common failure modes (divergence, overfit, OOM).

Do NOT discuss hyperparameters that are not being tuned. Do NOT \
exhaustively describe every line of code — stay focused on what an \
optimizer agent needs to make good proposals. Output plain text suitable \
for direct inclusion in a prompt.\
"""


SYSTEM_PROMPT = """\
You are a hyperparameter optimizer. You observe experiment results and \
propose configurations as instructed by the user's objective.

You see all prior trials in the context. Learn from both good and bad \
results.

Do NOT anchor to the initial configuration. In early iterations, test \
each parameter at diverse values across its full range to map out the \
search space before narrowing down.

When you detect a monotonic trend (e.g. lower X consistently helps), \
make a large jump (e.g. halve or double the value) rather than small \
increments. Bold moves are more informative than tiny perturbations.

Beware confounded conclusions: early results about one parameter may be \
wrong if other parameters were suboptimal at the time. Periodically \
re-test previous assumptions with your current best values for other \
parameters.

You MUST call propose_config with values for ALL parameters. Be concise \
— just call the tool.\
"""


def format_proposal(response: dict) -> str:
    """Render a raw provider response as a context entry.

    Uses THINKING/RESPONSE/PROPOSED section markers.

    Args:
        response: A dict in the ``Provider.propose`` return shape with
            keys ``reasoning``, ``content``, ``tool_args``.

    Returns:
        A multi-line string suitable for ``Context.add()``.
    """
    parts: list[str] = []
    if response.get("reasoning"):
        parts.append(f"THINKING:\n{response['reasoning']}")
    if response.get("content"):
        parts.append(f"RESPONSE:\n{response['content']}")
    parts.append(f"PROPOSED: {response.get('tool_args', {})}")
    return "\n".join(parts)


def format_result(cfg: dict, value: float, note: str | None) -> str:
    """Render a trial result as a context entry.

    Args:
        cfg: The hyperparameter configuration that was tried.
        value: The empirical metric the user reported.
        note: Optional free-form note (e.g. "diverged at step 1200").

    Returns:
        A multi-line string suitable for ``Context.add()``.
    """
    text = f"RESULT: value={value}\n  cfg={cfg}"
    if note:
        text += f"\n  note: {note}"
    return text


def build_user_message(
    context_text: str,
    space_summary: list[str],
    objective: str,
) -> str:
    """Build the user message sent to the LLM on every ``Tuner.suggest()``.

    Lays the message out as headed sections.

    Args:
        context_text: The full rendered context (from ``Context.render()``).
        space_summary: One short string per parameter (from
            ``Param.summary()``).
        objective: ``"minimize"`` or ``"maximize"``.

    Returns:
        The full string to send as the user message.
    """
    parts = [
        "## Task",
        f"Optimize hyperparameters. Objective: {objective}.",
        "",
        "## Parameter bounds",
    ]
    for s in space_summary:
        parts.append(f"  {s}")
    parts.extend(
        [
            "",
            "## Context",
            context_text if context_text else "(empty)",
            "",
            "Propose the next configuration.",
        ]
    )
    return "\n".join(parts)
