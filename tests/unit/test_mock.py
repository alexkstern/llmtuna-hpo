"""Tests for llmtuna.providers.mock — MockProvider."""

import pytest

from llmtuna.providers.base import Provider
from llmtuna.providers.mock import MockProvider


# ============================================================
# ABC inheritance
# ============================================================

def test_mock_provider_is_a_provider():
    """MockProvider must satisfy the Provider ABC contract — without this
    the inheritance has been silently broken."""
    p = MockProvider(responses=[])
    assert isinstance(p, Provider)


# ============================================================
# propose() — returning queued responses (bare tool_args canonicalized)
# ============================================================

def test_returns_queued_response_in_full_shape():
    """Bare tool_args input is promoted to the full response shape."""
    p = MockProvider(responses=[{"lr": 0.001}])
    result = p.propose(system="sys", user="usr", tool_spec={"name": "t"})
    assert result == {"reasoning": "", "content": "", "tool_args": {"lr": 0.001}}


def test_pops_responses_in_order():
    p = MockProvider(responses=[{"lr": 1}, {"lr": 2}, {"lr": 3}])
    assert p.propose(system="", user="", tool_spec={})["tool_args"] == {"lr": 1}
    assert p.propose(system="", user="", tool_spec={})["tool_args"] == {"lr": 2}
    assert p.propose(system="", user="", tool_spec={})["tool_args"] == {"lr": 3}


def test_response_dict_passes_through_unchanged_in_tool_args():
    """Complex nested structures inside tool_args pass through verbatim."""
    inner = {"lr": 0.001, "schedule": {"warmup": 100, "decay": "cosine"}}
    p = MockProvider(responses=[inner])
    result = p.propose(system="", user="", tool_spec={})
    assert result["tool_args"] == inner


# ============================================================
# propose() — full response shape preserved
# ============================================================

def test_full_response_shape_preserved():
    """Queued responses already in full shape are returned as-is."""
    full = {
        "reasoning": "I think lr=0.001 is a good start.",
        "content": "",
        "tool_args": {"lr": 0.001},
    }
    p = MockProvider(responses=[full])
    result = p.propose(system="", user="", tool_spec={})
    assert result == full


def test_full_response_shape_with_partial_keys_filled():
    """Full shape with only some optional keys gets defaults filled."""
    p = MockProvider(responses=[
        {"reasoning": "thinking...", "tool_args": {"lr": 0.5}}
    ])
    result = p.propose(system="", user="", tool_spec={})
    assert result == {
        "reasoning": "thinking...",
        "content": "",
        "tool_args": {"lr": 0.5},
    }


def test_full_response_shape_with_only_tool_args_not_double_wrapped():
    """Regression: a dict with only 'tool_args' key is full shape, not bare.
    Previously this was double-wrapped into ``{"tool_args": {"tool_args":
    {"lr": 0.001}}}`` due to a too-strict heuristic."""
    p = MockProvider(responses=[{"tool_args": {"lr": 0.001}}])
    result = p.propose(system="", user="", tool_spec={})
    assert result == {"reasoning": "", "content": "", "tool_args": {"lr": 0.001}}


# ============================================================
# propose() — recording call arguments
# ============================================================

def test_records_call_args():
    p = MockProvider(responses=[{"x": 1}])
    p.propose(system="sys", user="usr", tool_spec={"name": "tool"})
    assert len(p.calls) == 1
    assert p.calls[0] == {
        "system": "sys",
        "user": "usr",
        "tool_spec": {"name": "tool"},
    }


def test_records_multiple_calls_in_order():
    p = MockProvider(responses=[{"x": 1}, {"x": 2}])
    p.propose(system="s1", user="u1", tool_spec={"a": 1})
    p.propose(system="s2", user="u2", tool_spec={"a": 2})
    assert len(p.calls) == 2
    assert p.calls[0]["system"] == "s1"
    assert p.calls[1]["system"] == "s2"


# ============================================================
# Empty / exhausted queue
# ============================================================

def test_raises_when_out_of_responses():
    p = MockProvider(responses=[{"x": 1}])
    p.propose(system="", user="", tool_spec={})
    with pytest.raises(IndexError, match="out of responses"):
        p.propose(system="", user="", tool_spec={})


def test_raises_immediately_on_empty_queue():
    p = MockProvider(responses=[])
    with pytest.raises(IndexError, match="out of responses"):
        p.propose(system="", user="", tool_spec={})


# ============================================================
# Construction-time isolation
# ============================================================

def test_instances_are_independent():
    p1 = MockProvider(responses=[{"x": 1}])
    p2 = MockProvider(responses=[{"x": 2}])
    assert p1.propose(system="", user="", tool_spec={})["tool_args"] == {"x": 1}
    assert p2.propose(system="", user="", tool_spec={})["tool_args"] == {"x": 2}
    assert len(p1.calls) == 1
    assert len(p2.calls) == 1


def test_responses_list_is_copied_at_construction():
    """Mutating the input list after construction must not affect the queue."""
    queue = [{"x": 1}, {"x": 2}]
    p = MockProvider(responses=queue)
    queue.append({"x": 3})
    queue.clear()
    assert p.propose(system="", user="", tool_spec={})["tool_args"] == {"x": 1}
    assert p.propose(system="", user="", tool_spec={})["tool_args"] == {"x": 2}
    with pytest.raises(IndexError):
        p.propose(system="", user="", tool_spec={})


# ============================================================
# complete() — separate queue, separate call shape
# ============================================================

def test_complete_returns_queued_string():
    p = MockProvider(completion_responses=["hello world"])
    assert p.complete(system="sys", user="usr") == "hello world"


def test_complete_pops_in_order():
    p = MockProvider(completion_responses=["a", "b", "c"])
    assert p.complete(system="", user="") == "a"
    assert p.complete(system="", user="") == "b"
    assert p.complete(system="", user="") == "c"


def test_complete_records_call_with_max_tokens():
    p = MockProvider(completion_responses=["ok"])
    p.complete(system="sys", user="usr", max_tokens=512)
    assert len(p.calls) == 1
    assert p.calls[0] == {"system": "sys", "user": "usr", "max_tokens": 512}


def test_complete_records_call_without_max_tokens():
    p = MockProvider(completion_responses=["ok"])
    p.complete(system="sys", user="usr")
    assert p.calls[0]["max_tokens"] is None


def test_complete_raises_when_queue_empty():
    p = MockProvider(completion_responses=[])
    with pytest.raises(IndexError, match="completion_responses"):
        p.complete(system="", user="")


def test_propose_and_complete_use_independent_queues():
    """Mixing the two call types is fine — each pops from its own queue."""
    p = MockProvider(
        responses=[{"lr": 0.001}],
        completion_responses=["a summary"],
    )
    assert p.complete(system="", user="") == "a summary"
    assert p.propose(system="", user="", tool_spec={})["tool_args"] == {"lr": 0.001}
    # Both queues now exhausted
    with pytest.raises(IndexError):
        p.complete(system="", user="")
    with pytest.raises(IndexError):
        p.propose(system="", user="", tool_spec={})


def test_default_construction_with_no_args():
    """Both queue args are optional (callers may only need one type)."""
    p = MockProvider()
    assert p.responses == []
    assert p.completion_responses == []
