"""Tests for llmtuna.tuner — Tuner and Trial."""

import json

import pytest

import llmtuna as lt
from llmtuna.providers.mock import MockProvider
from llmtuna.tuner import Trial


# ============================================================
# Helpers
# ============================================================

def _basic_space():
    """Two-param space used by most suggest/observe tests."""
    return [
        lt.Float(name="lr", description="learning rate", bounds=(1e-6, 1.0)),
        lt.Int(name="depth", description="num layers", bounds=(1, 20)),
    ]


# ============================================================
# Construction
# ============================================================

def test_construction_rejects_empty_space():
    with pytest.raises(ValueError, match="at least one parameter"):
        lt.Tuner(provider=MockProvider(responses=[]), space=[])


def test_construction_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate parameter names"):
        lt.Tuner(
            provider=MockProvider(responses=[]),
            space=[
                lt.Float(name="lr", description="..."),
                lt.Float(name="lr", description="..."),
            ],
        )


def test_construction_rejects_bad_objective():
    with pytest.raises(ValueError, match="objective must be"):
        lt.Tuner(
            provider=MockProvider(responses=[]),
            space=_basic_space(),
            objective="banana",
        )


def test_construction_uses_defaults_when_none():
    """When formatter kwargs are None, the defaults from defaults.py are used."""
    from llmtuna import defaults as d

    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    assert opt.system_prompt == d.SYSTEM_PROMPT
    assert opt.format_proposal is d.format_proposal
    assert opt.format_result is d.format_result
    assert opt.build_user_message is d.build_user_message


def test_construction_uses_custom_overrides():
    custom_sys = "you are a tuner"
    custom_proposal = lambda r: "P"  # noqa: E731
    custom_result = lambda c, v, n: "R"  # noqa: E731
    custom_user = lambda ct, ss, ob: "U"  # noqa: E731

    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        system_prompt=custom_sys,
        format_proposal=custom_proposal,
        format_result=custom_result,
        build_user_message=custom_user,
    )
    assert opt.system_prompt == custom_sys
    assert opt.format_proposal is custom_proposal
    assert opt.format_result is custom_result
    assert opt.build_user_message is custom_user


def test_construction_creates_empty_context_and_history():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    assert len(opt.context) == 0
    assert opt.history == []


# ============================================================
# Tool spec building
# ============================================================

def test_tool_spec_has_all_params_required():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    spec = opt._tool_spec()
    assert spec["name"] == "propose_config"
    assert set(spec["parameters"]["properties"].keys()) == {"lr", "depth"}
    assert spec["parameters"]["required"] == ["lr", "depth"]


def test_tool_spec_property_types_match_param_schema():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    spec = opt._tool_spec()
    assert spec["parameters"]["properties"]["lr"]["type"] == "number"
    assert spec["parameters"]["properties"]["depth"]["type"] == "integer"


# ============================================================
# suggest()
# ============================================================

def test_suggest_returns_validated_cfg():
    p = MockProvider(responses=[{"lr": 0.001, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space())
    cfg = opt.suggest()
    assert cfg == {"lr": 0.001, "depth": 5}


def test_suggest_calls_provider_with_system_user_tool_spec():
    p = MockProvider(responses=[{"lr": 0.001, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space())
    opt.suggest()
    assert len(p.calls) == 1
    call = p.calls[0]
    assert call["system"] == opt.system_prompt
    assert "Optimize hyperparameters" in call["user"]
    assert call["tool_spec"]["name"] == "propose_config"


def test_suggest_user_message_includes_objective_and_space():
    p = MockProvider(responses=[{"lr": 0.001, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space(), objective="minimize")
    opt.suggest()
    user_msg = p.calls[0]["user"]
    assert "minimize" in user_msg
    assert "lr" in user_msg
    assert "depth" in user_msg


def test_suggest_appends_raw_response_to_context():
    response = {
        "reasoning": "I will start at lr=0.001",
        "content": "",
        "tool_args": {"lr": 0.001, "depth": 5},
    }
    p = MockProvider(responses=[response])
    opt = lt.Tuner(provider=p, space=_basic_space())
    opt.suggest()
    assert len(opt.context) == 1
    rendered = opt.context.entries[0].text
    assert "I will start at lr=0.001" in rendered
    assert "0.001" in rendered


def test_suggest_coerces_int_to_float_for_float_param():
    """Permissive coercion: LLM returns int 1 for a Float-typed param."""
    p = MockProvider(responses=[{"lr": 1, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space())
    cfg = opt.suggest()
    assert isinstance(cfg["lr"], float)
    assert cfg["lr"] == 1.0


def test_suggest_retries_on_validation_failure():
    """First response is out-of-bounds, second is valid — should retry once."""
    p = MockProvider(
        responses=[
            {"lr": 999.0, "depth": 5},  # lr above upper bound
            {"lr": 0.001, "depth": 5},  # valid
        ]
    )
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=3)
    cfg = opt.suggest()
    assert cfg == {"lr": 0.001, "depth": 5}
    assert len(p.calls) == 2


def test_suggest_retry_message_includes_previous_error():
    p = MockProvider(
        responses=[
            {"lr": 999.0, "depth": 5},
            {"lr": 0.001, "depth": 5},
        ]
    )
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=3)
    opt.suggest()
    second_user_msg = p.calls[1]["user"]
    assert "PREVIOUS ATTEMPT FAILED" in second_user_msg
    assert "outside bounds" in second_user_msg


def test_suggest_raises_after_max_retries_exhausted():
    p = MockProvider(
        responses=[
            {"lr": 999.0, "depth": 5},
            {"lr": 999.0, "depth": 5},
            {"lr": 999.0, "depth": 5},
        ]
    )
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=2)
    with pytest.raises(RuntimeError, match="validation failed on all 3 attempts"):
        opt.suggest()
    assert len(p.calls) == 3


def test_suggest_retries_on_extra_unknown_param():
    """First response includes a phantom param; retry then succeeds."""
    p = MockProvider(
        responses=[
            {"lr": 0.001, "depth": 5, "momentum": 0.9},  # extra "momentum"
            {"lr": 0.001, "depth": 5},                   # valid
        ]
    )
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=3)
    cfg = opt.suggest()
    assert cfg == {"lr": 0.001, "depth": 5}
    assert len(p.calls) == 2


def test_suggest_retry_message_lists_extra_params():
    p = MockProvider(
        responses=[
            {"lr": 0.001, "depth": 5, "momentum": 0.9, "weight_init": "kaiming"},
            {"lr": 0.001, "depth": 5},
        ]
    )
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=3)
    opt.suggest()
    second_user_msg = p.calls[1]["user"]
    assert "Unexpected parameters" in second_user_msg
    assert "momentum" in second_user_msg
    assert "weight_init" in second_user_msg


def test_suggest_raises_on_missing_required_param():
    p = MockProvider(
        responses=[
            {"lr": 0.001},  # depth missing
            {"lr": 0.001},
            {"lr": 0.001},
        ]
    )
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=2)
    with pytest.raises(RuntimeError, match="Missing required parameter 'depth'"):
        opt.suggest()


# ============================================================
# observe()
# ============================================================

def test_observe_appends_trial_to_history():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2)
    assert len(opt.history) == 1
    assert opt.history[0] == Trial(cfg={"lr": 0.001, "depth": 5}, value=4.2, note=None)


def test_observe_appends_formatted_result_to_context():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2)
    assert len(opt.context) == 1
    text = opt.context.entries[0].text
    assert "4.2" in text


def test_observe_records_optional_note():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.observe(
        cfg={"lr": 0.001, "depth": 5},
        value=4.2,
        note="diverged at step 1200",
    )
    assert opt.history[0].note == "diverged at step 1200"
    assert "diverged at step 1200" in opt.context.entries[0].text


def test_observe_default_note_is_none():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2)
    assert opt.history[0].note is None


# ============================================================
# best property
# ============================================================

def test_best_is_none_when_history_empty():
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    assert opt.best is None


def test_best_picks_minimum_when_minimize():
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        objective="minimize",
    )
    opt.observe(cfg={"lr": 0.1, "depth": 1}, value=5.0)
    opt.observe(cfg={"lr": 0.01, "depth": 5}, value=2.0)
    opt.observe(cfg={"lr": 0.001, "depth": 10}, value=3.0)
    assert opt.best == {"cfg": {"lr": 0.01, "depth": 5}, "value": 2.0}


def test_best_picks_maximum_when_maximize():
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        objective="maximize",
    )
    opt.observe(cfg={"lr": 0.1, "depth": 1}, value=5.0)
    opt.observe(cfg={"lr": 0.01, "depth": 5}, value=2.0)
    opt.observe(cfg={"lr": 0.001, "depth": 10}, value=3.0)
    assert opt.best == {"cfg": {"lr": 0.1, "depth": 1}, "value": 5.0}


def test_best_returns_first_on_tie():
    """Chronological order wins on ties — important for reproducibility."""
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        objective="minimize",
    )
    opt.observe(cfg={"lr": 0.1, "depth": 1}, value=2.0)
    opt.observe(cfg={"lr": 0.01, "depth": 5}, value=2.0)
    assert opt.best["cfg"]["depth"] == 1


# ============================================================
# render_prompt() and save_prompt()
# ============================================================

def test_render_prompt_returns_three_keys():
    p = MockProvider(responses=[])
    opt = lt.Tuner(provider=p, space=_basic_space())
    rp = opt.render_prompt()
    assert set(rp.keys()) == {"system", "user", "tool_spec"}


def test_render_prompt_makes_no_provider_calls():
    """Crucial — preview must not cost API credits."""
    p = MockProvider(responses=[])
    opt = lt.Tuner(provider=p, space=_basic_space())
    opt.render_prompt()
    opt.render_prompt()
    opt.render_prompt()
    assert p.calls == []


def test_render_prompt_matches_what_suggest_actually_sends():
    """The load-bearing parity test. If render_prompt drifts from what
    suggest sends, faithfulness is broken."""
    p = MockProvider(responses=[{"lr": 0.001, "depth": 5}])
    opt = lt.Tuner(
        provider=p,
        space=_basic_space(),
        system_prompt="custom system prompt",
    )
    opt.context.add(text="prior note from the scientist")

    expected = opt.render_prompt()
    opt.suggest()
    sent = p.calls[0]

    assert sent["system"] == expected["system"]
    assert sent["user"] == expected["user"]
    assert sent["tool_spec"] == expected["tool_spec"]


def test_render_prompt_output_is_json_serializable():
    p = MockProvider(responses=[])
    opt = lt.Tuner(provider=p, space=_basic_space())
    json.dumps(opt.render_prompt())


def test_save_prompt_writes_render_prompt_output(tmp_path):
    p = MockProvider(responses=[])
    opt = lt.Tuner(provider=p, space=_basic_space(), system_prompt="X")
    save_path = tmp_path / "prompt.json"
    opt.save_prompt(path=save_path)

    written = json.loads(save_path.read_text())
    assert written == opt.render_prompt()


# ============================================================
# system_prompt save / load round-trip
# ============================================================

def test_save_includes_system_prompt(tmp_path):
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        system_prompt="custom",
    )
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)
    data = json.loads(save_path.read_text())
    assert data["system_prompt"] == "custom"


def test_load_restores_system_prompt_without_kwarg(tmp_path):
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        system_prompt="custom",
    )
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)

    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    assert restored.system_prompt == "custom"


def test_load_kwarg_overrides_saved_system_prompt(tmp_path):
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        system_prompt="from-save",
    )
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)

    restored = lt.Tuner.load(
        path=save_path,
        provider=MockProvider(responses=[]),
        system_prompt="overridden-at-load",
    )
    assert restored.system_prompt == "overridden-at-load"


def test_load_old_save_without_system_prompt_falls_back_to_default(tmp_path):
    """Backward compat: pre-render_prompt save files lack system_prompt."""
    from llmtuna import defaults as d

    old_data = {
        "version": 1,
        "objective": "minimize",
        "max_retries": 3,
        "space": [
            {"__type__": "Float", "name": "lr", "description": "lr",
             "bounds": [0.0, 1.0], "initial": 0.001},
            {"__type__": "Int", "name": "depth", "description": "depth",
             "bounds": [1, 20], "initial": None},
        ],
        "context": {"entries": []},
        "history": [],
    }
    save_path = tmp_path / "old.json"
    save_path.write_text(json.dumps(old_data))

    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    assert restored.system_prompt == d.SYSTEM_PROMPT


def test_render_prompt_after_save_load_is_identical(tmp_path):
    """The audit guarantee: a saved run reloads to the same prompt."""
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        system_prompt="custom for audit",
    )
    opt.context.add(text="prior note")
    opt.observe(cfg={"lr": 0.5, "depth": 5}, value=0.42)

    pre = opt.render_prompt()
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)

    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    post = restored.render_prompt()

    assert pre == post


# ============================================================
# save / load round-trip
# ============================================================

def test_save_writes_valid_json(tmp_path):
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2)
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)
    data = json.loads(save_path.read_text())
    assert data["version"] == 1
    assert data["objective"] == "minimize"
    assert len(data["history"]) == 1


def test_save_does_not_serialize_provider(tmp_path):
    """API keys must never hit disk."""
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)
    text = save_path.read_text()
    assert "MockProvider" not in text
    assert "api_key" not in text.lower()


def test_load_reconstructs_empty_state(tmp_path):
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        objective="maximize",
    )
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)

    fresh = MockProvider(responses=[])
    restored = lt.Tuner.load(path=save_path, provider=fresh)
    assert restored.objective == "maximize"
    assert restored.provider is fresh
    assert len(restored.history) == 0
    assert len(restored.context) == 0
    assert [p.name for p in restored.space] == ["lr", "depth"]


def test_load_round_trips_history(tmp_path):
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2, note="ok")
    opt.observe(cfg={"lr": 0.01, "depth": 8}, value=3.5)
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)

    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    assert len(restored.history) == 2
    assert restored.history[0] == Trial(
        cfg={"lr": 0.001, "depth": 5}, value=4.2, note="ok"
    )
    assert restored.history[1].note is None
    assert restored.best == {"cfg": {"lr": 0.01, "depth": 8}, "value": 3.5}


def test_load_round_trips_context(tmp_path):
    opt = lt.Tuner(provider=MockProvider(responses=[]), space=_basic_space())
    opt.context.add(text="architecture: GPT-2 with RoPE")
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2)
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)

    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    assert restored.context.render() == opt.context.render()


def test_load_without_overrides_restores_system_prompt_but_not_formatters(tmp_path):
    """system_prompt IS serialized and restored automatically.
    Custom formatters are NOT (functions can't round-trip through JSON);
    they fall back to defaults unless re-passed at load time."""
    from llmtuna import defaults as d

    original = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        system_prompt="custom prompt",
        format_proposal=lambda r: "CUSTOM",
    )
    save_path = tmp_path / "state.json"
    original.save(path=save_path)

    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    assert restored.system_prompt == "custom prompt"   # restored from save file
    assert restored.format_proposal is d.format_proposal   # not serializable; default


def test_load_with_overrides_applies_them(tmp_path):
    """Re-passing custom formatters at load() time restores them."""
    custom_sys = "custom prompt v2"
    custom_proposal = lambda r: "CUSTOM_AT_LOAD"  # noqa: E731

    original = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
    )
    save_path = tmp_path / "state.json"
    original.save(path=save_path)

    restored = lt.Tuner.load(
        path=save_path,
        provider=MockProvider(responses=[]),
        system_prompt=custom_sys,
        format_proposal=custom_proposal,
    )
    assert restored.system_prompt == custom_sys
    assert restored.format_proposal is custom_proposal


def test_load_round_trips_space_with_choice(tmp_path):
    """Param subtype info must survive the round trip."""
    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=[
            lt.Float(name="lr", description="lr", bounds=(1e-6, 1.0)),
            lt.Int(name="depth", description="depth", bounds=(1, 20)),
            lt.Choice(name="act", description="activation", options=["relu", "gelu"]),
        ],
    )
    save_path = tmp_path / "state.json"
    opt.save(path=save_path)
    restored = lt.Tuner.load(path=save_path, provider=MockProvider(responses=[]))
    assert isinstance(restored.space[0], lt.Float)
    assert isinstance(restored.space[1], lt.Int)
    assert isinstance(restored.space[2], lt.Choice)
    assert restored.space[0].bounds == (1e-6, 1.0)
    assert restored.space[2].options == ["relu", "gelu"]


# ============================================================
# Custom formatter overrides are actually invoked
# ============================================================

def test_custom_format_proposal_is_called():
    calls = []

    def my_fmt(response):
        calls.append(response)
        return "CUSTOM_PROPOSAL_OUTPUT"

    p = MockProvider(responses=[{"lr": 0.001, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space(), format_proposal=my_fmt)
    opt.suggest()
    assert len(calls) == 1
    assert opt.context.entries[0].text == "CUSTOM_PROPOSAL_OUTPUT\n\n"


def test_custom_format_result_is_called():
    calls = []

    def my_fmt(cfg, value, note):
        calls.append((cfg, value, note))
        return "CUSTOM_RESULT_OUTPUT"

    opt = lt.Tuner(
        provider=MockProvider(responses=[]),
        space=_basic_space(),
        format_result=my_fmt,
    )
    opt.observe(cfg={"lr": 0.001, "depth": 5}, value=4.2)
    assert len(calls) == 1
    assert opt.context.entries[0].text == "CUSTOM_RESULT_OUTPUT\n\n"


def test_custom_build_user_message_is_called():
    calls = []

    def my_builder(context_text, space_summary, objective):
        calls.append((context_text, space_summary, objective))
        return "CUSTOM_USER_MESSAGE"

    p = MockProvider(responses=[{"lr": 0.001, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space(), build_user_message=my_builder)
    opt.suggest()
    assert len(calls) == 1
    assert p.calls[0]["user"] == "CUSTOM_USER_MESSAGE"


def test_suggest_retries_malformed_provider_output():
    class FlakyProvider(MockProvider):
        def propose(self, system, user, tool_spec):
            if not self.calls:
                self.calls.append({"system": system, "user": user, "tool_spec": tool_spec})
                raise ValueError("OpenRouter: tool arguments not valid JSON: '{\"lr\": '")
            return super().propose(system, user, tool_spec)

    p = FlakyProvider(responses=[{"lr": 0.01, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=2)
    assert opt.suggest() == {"lr": 0.01, "depth": 5}
    assert "not valid JSON" in p.calls[1]["user"]


def test_suggest_retries_non_object_tool_args():
    p = MockProvider(responses=[{"tool_args": None}, {"lr": 0.01, "depth": 5}])
    opt = lt.Tuner(provider=p, space=_basic_space(), max_retries=2)
    assert opt.suggest() == {"lr": 0.01, "depth": 5}
    assert "must be a JSON object" in p.calls[1]["user"]
