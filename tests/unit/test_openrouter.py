"""Tests for llmtuna.providers.openrouter — OpenRouter (mock-backed).

These are unit tests: we inject a fake OpenAI client into the OpenRouter
instance via ``p._client = FakeOpenAIClient(...)`` and feed it canned
``ChatCompletion``-shaped responses built with ``SimpleNamespace``. No
network calls happen here. Live-API tests live in ``tests/integration/``.
"""

import json
from types import SimpleNamespace

import pytest

from llmtuna.providers.base import Provider
from llmtuna.providers.openrouter import OpenRouter


# ============================================================
# Fakes mimicking the OpenAI SDK shape
# ============================================================

def _make_message(
    content: str = "",
    tool_args=None,
    tool_name: str = "propose_config",
    reasoning: str | None = None,
    reasoning_details=None,
):
    """Build a SimpleNamespace mimicking an OpenAI ChatCompletion message.

    ``tool_args`` may be a dict (will be JSON-encoded into ``arguments``)
    or a raw string (used verbatim — useful for testing bad-JSON paths).
    """
    tool_calls = None
    if tool_args is not None:
        arguments = (
            json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
        )
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(name=tool_name, arguments=arguments),
            )
        ]
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        msg.reasoning = reasoning
    if reasoning_details is not None:
        msg.reasoning_details = reasoning_details
    return msg


def _make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_empty_response():
    """No choices — simulates a transient empty completion."""
    return SimpleNamespace(choices=[])


class _FakeChatCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise IndexError("FakeChatCompletions out of responses")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class _FakeOpenAIClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))

    @property
    def calls(self):
        return self.chat.completions.calls


def _make_provider(responses, **kwargs):
    """Construct an OpenRouter with ``api_key="test"`` and a fake client."""
    p = OpenRouter(model="test/model", api_key="test", **kwargs)
    p._client = _FakeOpenAIClient(responses)
    return p


_BASIC_TOOL_SPEC = {
    "name": "propose_config",
    "description": "Propose a config",
    "parameters": {
        "type": "object",
        "properties": {"lr": {"type": "number"}},
        "required": ["lr"],
    },
}


# ============================================================
# ABC inheritance
# ============================================================

def test_openrouter_is_a_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    p = OpenRouter(model="m")
    assert isinstance(p, Provider)


# ============================================================
# Construction
# ============================================================

def test_construction_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    p = OpenRouter(model="m")
    assert p.model == "m"


def test_construction_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no API key"):
        OpenRouter(model="m")


def test_construction_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    p = OpenRouter(model="m", api_key="explicit")
    assert p.model == "m"


def test_construction_rejects_force_tool_with_thinking():
    """force_tool=True with a thinking budget is rejected at construction."""
    with pytest.raises(ValueError, match="force_tool=True cannot be combined"):
        OpenRouter(
            model="m",
            api_key="test",
            force_tool=True,
            thinking_budget=1024,
        )


def test_construction_accepts_force_tool_with_thinking_disabled():
    """The valid combo: forced tool calls AND no wasted reasoning budget."""
    p = OpenRouter(
        model="m",
        api_key="test",
        force_tool=True,
        thinking_budget=0,
    )
    assert p.force_tool is True
    assert p.thinking_budget == 0


def test_construction_stores_all_knobs():
    p = OpenRouter(
        model="anthropic/claude-haiku-4-5",
        api_key="test",
        thinking_budget=5000,
        max_tokens=512,
        max_retries=5,
        extra_body={"top_p": 0.9},
        base_url="https://example.com/v1",
    )
    assert p.model == "anthropic/claude-haiku-4-5"
    assert p.thinking_budget == 5000
    assert p.max_tokens == 512
    assert p.max_retries == 5
    assert p.extra_body == {"top_p": 0.9}
    assert p.base_url == "https://example.com/v1"


# ============================================================
# Request building
# ============================================================

def test_propose_builds_messages_with_system_and_user():
    p = _make_provider([_make_response(_make_message(tool_args={"lr": 0.001}))])
    p.propose(system="SYS", user="USR", tool_spec=_BASIC_TOOL_SPEC)
    call = p._client.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]


def test_propose_wraps_tool_spec_in_openai_function_format():
    p = _make_provider([_make_response(_make_message(tool_args={"lr": 0.001}))])
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    call = p._client.calls[0]
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "propose_config",
                "description": "Propose a config",
                "parameters": _BASIC_TOOL_SPEC["parameters"],
            },
        }
    ]


def test_propose_does_not_send_tool_choice_by_default():
    """Default behavior: rely on system-prompt forcing, preserve reasoning.
    Forcing tool_choice cannot be combined with extended thinking."""
    p = _make_provider([_make_response(_make_message(tool_args={"lr": 0.001}))])
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    assert "tool_choice" not in p._client.calls[0]


def test_propose_sends_tool_choice_when_force_tool_true():
    """Opt-in belt-and-suspenders mode for weaker models. Must set
    thinking_budget=0 — the reasoning vs. forcing trade-off is
    enforced at construction (see test_construction_rejects_force_tool_with_thinking)."""
    p = _make_provider(
        [_make_response(_make_message(tool_args={"lr": 0.001}))],
        force_tool=True,
        thinking_budget=0,
    )
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    assert p._client.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "propose_config"},
    }


def test_propose_includes_reasoning_in_extra_body_when_thinking_enabled():
    p = _make_provider(
        [_make_response(_make_message(tool_args={"lr": 0.001}))],
        thinking_budget=8000,
    )
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    call = p._client.calls[0]
    assert call["extra_body"] == {"reasoning": {"max_tokens": 8000}}


def test_propose_omits_extra_body_when_thinking_disabled_and_no_user_extras():
    p = _make_provider(
        [_make_response(_make_message(tool_args={"lr": 0.001}))],
        thinking_budget=0,
    )
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    call = p._client.calls[0]
    assert "extra_body" not in call


def test_propose_merges_user_extra_body_with_user_keys_winning():
    p = _make_provider(
        [_make_response(_make_message(tool_args={"lr": 0.001}))],
        thinking_budget=8000,
        extra_body={"reasoning": {"max_tokens": 1234}, "top_p": 0.9},
    )
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    call = p._client.calls[0]
    assert call["extra_body"] == {
        "reasoning": {"max_tokens": 1234},
        "top_p": 0.9,
    }


def test_propose_passes_model_and_max_tokens():
    p = _make_provider(
        [_make_response(_make_message(tool_args={"lr": 0.001}))],
        max_tokens=1234,
    )
    p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    call = p._client.calls[0]
    assert call["model"] == "test/model"
    assert call["max_tokens"] == 1234


# ============================================================
# Response shape
# ============================================================

def test_propose_returns_full_shape():
    msg = _make_message(
        content="some text",
        tool_args={"lr": 0.001, "depth": 5},
        reasoning="step by step thinking",
    )
    p = _make_provider([_make_response(msg)])
    result = p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    assert result == {
        "reasoning": "step by step thinking",
        "content": "some text",
        "tool_args": {"lr": 0.001, "depth": 5},
    }


# ============================================================
# Retry logic
# ============================================================

def test_propose_retries_on_empty_choices():
    valid = _make_response(_make_message(tool_args={"lr": 0.001}))
    p = _make_provider(
        [_make_empty_response(), _make_empty_response(), valid],
        max_retries=3,
    )
    result = p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)
    assert result["tool_args"] == {"lr": 0.001}
    assert len(p._client.calls) == 3


def test_propose_raises_when_all_retries_return_empty():
    p = _make_provider(
        [_make_empty_response(), _make_empty_response(), _make_empty_response()],
        max_retries=3,
    )
    with pytest.raises(RuntimeError, match="no choices"):
        p.propose(system="", user="", tool_spec=_BASIC_TOOL_SPEC)


# ============================================================
# _parse_message
# ============================================================

def test_parse_message_extracts_content():
    msg = _make_message(content="hello world", tool_args={"lr": 0.001})
    p = _make_provider([])
    assert p._parse_message(msg)["content"] == "hello world"


def test_parse_message_extracts_reasoning_from_reasoning_field():
    msg = _make_message(tool_args={"lr": 0.001}, reasoning="raw reasoning text")
    p = _make_provider([])
    assert p._parse_message(msg)["reasoning"] == "raw reasoning text"


def test_parse_message_falls_back_to_reasoning_details_namespace():
    """When msg.reasoning is empty, concatenate reasoning_details items."""
    msg = _make_message(
        tool_args={"lr": 0.001},
        reasoning="",
        reasoning_details=[
            SimpleNamespace(text="part one. "),
            SimpleNamespace(text="part two."),
        ],
    )
    p = _make_provider([])
    assert p._parse_message(msg)["reasoning"] == "part one. part two."


def test_parse_message_reasoning_details_from_dicts():
    """OpenRouter sometimes returns reasoning_details as dicts."""
    msg = _make_message(
        tool_args={"lr": 0.001},
        reasoning="",
        reasoning_details=[
            {"type": "reasoning.text", "text": "thinking..."},
            {"type": "reasoning.text", "text": " more thinking."},
        ],
    )
    p = _make_provider([])
    assert p._parse_message(msg)["reasoning"] == "thinking... more thinking."


def test_parse_message_tool_args_parsed():
    msg = _make_message(tool_args={"lr": 1e-3, "depth": 12})
    p = _make_provider([])
    assert p._parse_message(msg)["tool_args"] == {"lr": 1e-3, "depth": 12}


def test_parse_message_returns_empty_tool_args_when_no_tool_call():
    """When the model returns only text (no tool call), we return empty
    tool_args rather than raising — the Tuner's validation retry then
    re-prompts naturally instead of crashing the loop."""
    msg = _make_message(
        content="I refuse to call the tool",
        tool_args=None,
        reasoning="Some reasoning here",
    )
    p = _make_provider([])
    result = p._parse_message(msg)
    assert result["tool_args"] == {}
    assert result["reasoning"] == "Some reasoning here"
    assert result["content"] == "I refuse to call the tool"


def test_parse_message_handles_reasoning_encrypted():
    """Some OpenRouter routes return reasoning as encrypted blocks.
    Surface as a placeholder rather than silently dropping."""
    msg = _make_message(
        tool_args={"lr": 0.001},
        reasoning="",
        reasoning_details=[
            {"type": "reasoning.text", "text": "first part. "},
            {"type": "reasoning.encrypted", "data": "opaque-blob"},
            {"type": "reasoning.text", "text": " final part."},
        ],
    )
    p = _make_provider([])
    out = p._parse_message(msg)["reasoning"]
    assert "first part." in out
    assert "[reasoning encrypted by provider]" in out
    assert "final part." in out


def test_parse_message_raises_on_invalid_json():
    """Tool arguments string is not valid JSON."""
    msg = _make_message(tool_args="not valid json {{{")
    p = _make_provider([])
    with pytest.raises(ValueError, match="not valid JSON"):
        p._parse_message(msg)


# ============================================================
# complete() — free-form text generation
# ============================================================

def _make_text_response(content: str):
    """Build a SimpleNamespace mimicking a non-tool ChatCompletion response."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_complete_returns_message_content():
    p = _make_provider([_make_text_response("a brief summary")])
    assert p.complete(system="sys", user="usr") == "a brief summary"


def test_complete_passes_max_tokens_when_given():
    p = _make_provider([_make_text_response("ok")])
    p.complete(system="", user="", max_tokens=512)
    assert p._client.calls[0]["max_tokens"] == 512


def test_complete_falls_back_to_provider_max_tokens():
    p = _make_provider([_make_text_response("ok")], max_tokens=4321)
    p.complete(system="", user="")
    assert p._client.calls[0]["max_tokens"] == 4321


def test_complete_does_not_send_tool_or_extra_body():
    """complete() is plain text generation — no tool spec, no reasoning."""
    p = _make_provider([_make_text_response("ok")], thinking_budget=8000)
    p.complete(system="", user="")
    call = p._client.calls[0]
    assert "tools" not in call
    assert "tool_choice" not in call
    assert "extra_body" not in call


def test_complete_returns_empty_string_when_no_choices():
    p = _make_provider([_make_empty_response()])
    assert p.complete(system="", user="") == ""


def test_complete_returns_empty_string_when_content_is_none():
    msg = SimpleNamespace(content=None, tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    p = _make_provider([response])
    assert p.complete(system="", user="") == ""
