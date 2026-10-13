"""Integration tests for OpenRouter — hits the real API.

Skipped automatically when ``OPENROUTER_API_KEY`` is not set. Uses
cheap Haiku-4.5 with reasoning disabled and ``max_tokens`` capped, so
each test costs roughly $0.001. Run explicitly via:

    uv run pytest tests/integration/

These are NOT part of the default fast unit suite (``pyproject.toml``
sets ``testpaths = ["tests/unit"]``).
"""

import os

import pytest

import llmtuna as lt

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="needs OPENROUTER_API_KEY",
)

CHEAP_MODEL = "anthropic/claude-haiku-4-5"


def _cheap_provider():
    """OpenRouter configured for minimal-cost integration tests."""
    return lt.OpenRouter(
        model=CHEAP_MODEL,
        thinking_budget=0,
        max_tokens=300,
    )


def test_real_call_returns_validated_cfg():
    opt = lt.Tuner(
        provider=_cheap_provider(),
        space=[
            lt.Float(
                name="lr",
                description="AdamW learning rate",
                bounds=(1e-6, 1.0),
                initial=1e-3,
            ),
        ],
    )
    cfg = opt.suggest()
    assert "lr" in cfg
    assert isinstance(cfg["lr"], float)
    assert 1e-6 <= cfg["lr"] <= 1.0


def test_real_loop_two_trials_records_history_and_best():
    opt = lt.Tuner(
        provider=_cheap_provider(),
        space=[
            lt.Float(
                name="lr",
                description="learning rate",
                bounds=(0.0, 1.0),
                initial=0.5,
            ),
        ],
        objective="minimize",
    )
    for _ in range(2):
        cfg = opt.suggest()
        loss = (cfg["lr"] - 0.3) ** 2
        opt.observe(cfg=cfg, value=loss)

    assert len(opt.history) == 2
    assert opt.best is not None
    assert 0.0 <= opt.best["cfg"]["lr"] <= 1.0


def test_real_call_appends_full_response_to_context():
    """The provider's full response (including the proposal) reaches context."""
    opt = lt.Tuner(
        provider=_cheap_provider(),
        space=[
            lt.Int(
                name="depth",
                description="num transformer layers",
                bounds=(1, 20),
                initial=6,
            ),
        ],
    )
    opt.suggest()
    assert len(opt.context) == 1
    rendered = opt.context.render()
    assert "PROPOSED" in rendered or "depth" in rendered
