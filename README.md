# llmtuna

LLM-driven hyperparameter optimization for anything with knobs and a
scalar metric.

`llmtuna` is a general-purpose hyperparameter optimizer: you describe
a search space, point it at any system you can run and measure, and a
large language model proposes configurations to try.

The LLM sees a rolling text context — whatever notes, code, papers, or
prior trial results you choose to feed it — and reasons over them to
suggest the next configuration.

Inspired by work on LLMs-as-optimizers
([Yang et al., 2023](https://arxiv.org/abs/2309.03409);
[Jiang & Zhao, 2026](https://www.weco.ai/blog/autoresearch-vs-classical-hpo))
and Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).
`llmtuna` is purpose-built for the *config-in / scalar-out* loop — every
iteration produces a hyperparameter configuration evaluated against a
scalar metric. For more open-ended LLM-driven research workflows
(generate code, run experiments, change strategies), autoresearch is
the better fit.

## Why this exists

Classical samplers (Optuna's TPE, Ax/BoTorch, CARBS, hyperopt) are
oblivious to domain knowledge, prior literature, or anything the
scientist knows but hasn't encoded as a numerical prior. An LLM can
read a file describing your system, look at related results, and
incorporate that into proposals directly.

The library is intentionally minimal: three hyperparameter types
(`Float`, `Int`, `Choice`), one rolling text context, one provider
abstraction, one `Tuner` class.

## Install

Add `llmtuna` as a git dependency in your project's `pyproject.toml`:

```toml
[project]
dependencies = ["llmtuna"]

[tool.uv.sources]
llmtuna = { git = "https://github.com/alexkstern/llmtuna-hpo.git", rev = "v0.1.0" }
```

Or with [uv](https://github.com/astral-sh/uv) directly:

```bash
uv add "llmtuna @ git+https://github.com/alexkstern/llmtuna-hpo.git@v0.1.0"
```

Then in your code:

```python
import llmtuna as lt

opt = lt.Tuner(...)
```

To update later, bump the `rev` (e.g. `v0.0.2`) and run `uv sync`. Use
a commit hash in `rev` if you want even tighter pinning.

### Editable local clone (for hacking on llmtuna itself)

If you want to modify `llmtuna` while integrating it into your project:

```bash
git clone git@github.com:alexkstern/llmtuna-hpo.git ~/code/llmtuna

# from inside your own project's directory:
uv add --editable ~/code/llmtuna
```

This replaces the git-pinned source with a local editable install.
Switch back by re-running the `uv add "llmtuna @ git+..."` command.

## Quickstart

```python
import llmtuna as lt

opt = lt.Tuner(
    provider=lt.OpenRouter(model="anthropic/claude-sonnet-4-6"),
    space=[
        lt.Float(name="learning_rate",  description="learning rate",
                 bounds=(1e-6, 1.0), initial=1e-3),
        lt.Int(name="n_layers",         description="model depth",
               bounds=(1, 20)),
        lt.Float(name="regularization", description="weight decay / penalty strength",
                 bounds=(0.0, 0.1), initial=0.01),
    ],
    objective="minimize",
)

# Tell the LLM what you are doing
opt.context.add(
    text="You are optimizing the hyperparameters for a small ML pipeline "
         "that classifies sales records into 12 categories on ~50k rows of "
         "tabular features. Lower validation loss is better."
)

# Hand it the actual code, with short labels so it knows what each file is
opt.context.add(text="This is the training script:")
opt.context.add_file(path="train.py")
opt.context.add(text="This is the data loader:")
opt.context.add_file(path="dataloader.py")
opt.context.add(text="This is the model definition:")
opt.context.add_file(path="model.py")

for _ in range(30):
    cfg = opt.suggest()
    # e.g. cfg = {"learning_rate": 0.003, "n_layers": 8, "regularization": 0.01}
    val_loss = train_and_eval(**cfg)
    # e.g. val_loss = 0.482
    opt.observe(cfg=cfg, value=val_loss)

print(opt.best)         # {"cfg": {...}, "value": ...}
print(opt.history)      # list of all Trials in chronological order
opt.save(path="run.json")
```

## Concepts

### `Tuner`

The main class. Holds a `Provider`, a list of `Param` definitions
(the search space), an `objective` (`"minimize"` or `"maximize"`),
and a rolling `Context`. Exposes:

- `suggest() -> dict` — ask the LLM for the next configuration. Validates
  the LLM's response against the search space and retries up to
  `max_retries` times on validation failure (out-of-bounds, wrong type,
  missing or extra params), feeding the error back to the LLM each retry.
- `observe(cfg, value, note=None)` — record a trial result. Appends to
  `history` and to `context` so the LLM sees it on the next `suggest()`.
- `best` — property returning `{"cfg": ..., "value": ...}` for the
  winning trial (sign-aware on `objective`), or `None` if no trials yet.
- `history` — list of `Trial(cfg, value, note)` records.
- `context` — the `Context` object (see below).
- `save(path)` / `Tuner.load(path, provider=..., system_prompt=..., format_*=...)` —
  JSON serialization. The provider is never serialized (no API keys on
  disk); the user supplies a fresh provider on load. `system_prompt`
  IS serialized and restored automatically. Custom formatters must be
  re-passed at load time or they revert to defaults.
- `render_prompt() -> dict` — returns the exact `{"system", "user",
  "tool_spec"}` that `suggest()` would send next, **without making an
  API call**. Use it to preview / debug what the LLM is about to see.
- `save_prompt(path)` — writes `render_prompt()`'s output to a JSON
  file for human inspection. Frozen snapshot, NOT loadable — use
  `save(path)` for state round-trip.

### Hyperparameter types

```python
lt.Float(name, description, bounds=None, initial=None)
lt.Int(name, description, bounds=None, initial=None)
lt.Choice(name, description, options, initial=None)
```

`bounds` and `initial` are optional. The LLM is told the range and any
initial value via the tool's parameter description; if `bounds` is set,
returned values are validated and out-of-range proposals trigger a
retry.

### `Context`

A first-class, ordered text log shown to the LLM on every `suggest()`.
You add free-form text or snapshot files at any point — before the loop
or in between trials:

```python
opt.context.add(text="prior runs showed instability when learning_rate > 1e-2")
opt.context.add_file(path="model.py")     # reads the file NOW; stores its contents
```

Files are snapshotted at the time of the call. If you've since edited a
file on disk and want the LLM to see the new contents on the next
`suggest()`, call `opt.context.refresh()` (re-reads all file-backed
entries) or `opt.context.refresh(path="model.py")` (just one). Refresh
updates the stored text in place — it doesn't add or remove entries.

You can also prune the context — useful if a trial result was buggy or
you want to reset between phases:

```python
opt.context.pop(index=-1)   # remove the most recent entry
opt.context.pop(index=3)    # remove a specific one
opt.context.clear()         # remove everything
```

For long runs where dumping raw source files into context would blow
up the window, use `add_summary()` instead — it makes one upfront LLM
call (via `provider.complete()`) to digest the files into a focused
summary:

```python
opt.context.add_summary(
    provider=opt.provider,
    paths=["model.py", "train.py", "data.py"],
    hparam_names=["learning_rate", "n_layers", "regularization"],
    max_tokens=1000,        # cap the summary length
)
```

The summary is a normal text entry (not file-backed), so `refresh()`
ignores it. To re-summarize, `pop()` the old summary and call
`add_summary()` again.

The Tuner also auto-appends the LLM's full response (reasoning + content
+ proposed config) on every `suggest()`, and the trial result on every
`observe()` — so the LLM sees its own past thinking and the empirical
outcomes.

### `Provider`

The abstraction over LLM backends. Ships with one concrete provider:

```python
lt.OpenRouter(
    model="anthropic/claude-sonnet-4-6",
    api_key=None,             # falls back to OPENROUTER_API_KEY env var
    thinking_budget=10000,    # reasoning tokens; 0 disables
    max_tokens=2000,
    max_retries=3,            # transient empty-response retries
    force_tool=False,         # see "tool forcing vs reasoning" below
    extra_body=None,          # passthrough for vendor-specific knobs
    base_url=None,            # override to point at any OpenAI-compatible endpoint
)
```

OpenRouter speaks the OpenAI Chat Completions API, so most providers
(Anthropic, OpenAI, Google, Mistral, vLLM endpoints, local llama.cpp /
Ollama, etc.) are reachable through it — pass a custom `base_url=` for
a non-OpenRouter endpoint. Vendor SDK exceptions propagate unchanged;
`llmtuna` does not heuristically reclassify them.

**Tool forcing vs. reasoning.** Forcing the LLM to call `propose_config`
(OpenAI-style `tool_choice`) cannot be combined with extended thinking;
Anthropic's API, for one, rejects forced tool use when thinking is on.
You can have one or the other, not both. Two valid configurations:

| | `force_tool=False` (default) | `force_tool=True` |
|---|---|---|
| Tool call enforcement | system-prompt instruction ("You MUST call propose_config") | API-level `tool_choice` |
| Reasoning captured? | ✅ yes (when `thinking_budget > 0`) | ❌ off (requires `thinking_budget=0`) |
| When to use | default; capable models honor the instruction reliably | weaker models that ignore "you MUST" prompts; requires `thinking_budget=0` |

`OpenRouter(force_tool=True, thinking_budget>0)` is rejected at
construction with a `ValueError` rather than silently wasting credits.

To add a custom backend, subclass `Provider` and implement two methods:

```python
from llmtuna.providers.base import Provider

class MyBackend(Provider):
    def propose(self, system: str, user: str, tool_spec: dict) -> dict:
        # Forced tool call — used by Tuner.suggest() for hparam proposals.
        return {"reasoning": "...", "content": "...", "tool_args": {...}}

    def complete(self, system: str, user: str, *, max_tokens=None) -> str:
        # Plain text generation — used by Context.add_summary() and any
        # other free-form text helper.
        return "..."
```

### Customizing prompts and formatters

The system prompt and the three formatter functions (`format_proposal`,
`format_result`, `build_user_message`) are overridable on the `Tuner`:

```python
opt = lt.Tuner(
    provider=...,
    space=[...],
    system_prompt="You are an expert hyperparameter tuner for this domain...",
    format_result=lambda cfg, value, note: f"trial: {cfg} -> {value}",
)
```

Defaults live in `llmtuna/defaults.py` — read them as the spec.

## Citation

`llmtuna` was released with the paper *Rethinking Pre-pretraining*. A
citation will be added here once the paper is on arXiv.
