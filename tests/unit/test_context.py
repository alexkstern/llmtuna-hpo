"""Tests for llmtuna.context — Entry and Context."""

import json

import pytest

from llmtuna import defaults
from llmtuna.context import Context
from llmtuna.providers.mock import MockProvider


@pytest.fixture
def make_file(tmp_path):
    """Factory: write a file under ``tmp_path`` and return its Path."""

    def _make(name: str, body: str):
        p = tmp_path / name
        p.write_text(body)
        return p

    return _make


# ============================================================
# Construction
# ============================================================

def test_context_starts_empty():
    ctx = Context()
    assert len(ctx) == 0
    assert ctx.entries == []


# ============================================================
# add()
# ============================================================

def test_add_appends_one_entry():
    ctx = Context()
    ctx.add(text="hello")
    assert len(ctx) == 1


def test_add_preserves_order():
    ctx = Context()
    ctx.add(text="first")
    ctx.add(text="second")
    ctx.add(text="third")
    assert [e.text for e in ctx.entries] == ["first\n\n", "second\n\n", "third\n\n"]


def test_add_appends_separator_to_stored_text():
    ctx = Context()
    ctx.add(text="hello")
    assert ctx.entries[0].text == "hello\n\n"


def test_add_empty_string_stores_just_separator():
    ctx = Context()
    ctx.add(text="")
    assert ctx.entries[0].text == "\n\n"


def test_add_does_not_set_path():
    ctx = Context()
    ctx.add(text="note")
    assert ctx.entries[0].path is None


# ============================================================
# add_file()
# ============================================================

def test_add_file_reads_contents(make_file):
    f = make_file("note.txt", "file body")
    ctx = Context()
    ctx.add_file(path=f)
    assert ctx.entries[0].text == "file body\n\n"


def test_add_file_stores_path(make_file):
    f = make_file("note.txt", "body")
    ctx = Context()
    ctx.add_file(path=f)
    assert ctx.entries[0].path == f


def test_add_file_accepts_string_path(make_file):
    f = make_file("note.txt", "body")
    ctx = Context()
    ctx.add_file(path=str(f))
    assert ctx.entries[0].text == "body\n\n"


def test_add_file_raises_on_missing_file(tmp_path):
    ctx = Context()
    with pytest.raises(FileNotFoundError):
        ctx.add_file(path=tmp_path / "does-not-exist.txt")


def test_add_file_snapshots_contents(make_file):
    """Later edits to the file must not change the already-stored entry."""
    f = make_file("note.txt", "original")
    ctx = Context()
    ctx.add_file(path=f)
    f.write_text("modified")
    assert ctx.entries[0].text == "original\n\n"


# ============================================================
# refresh()
# ============================================================

def test_refresh_no_args_rereads_all_files(make_file):
    f1 = make_file("a.txt", "a1")
    f2 = make_file("b.txt", "b1")
    ctx = Context()
    ctx.add_file(path=f1)
    ctx.add_file(path=f2)
    f1.write_text("a2")
    f2.write_text("b2")
    count = ctx.refresh()
    assert count == 2
    assert ctx.entries[0].text == "a2\n\n"
    assert ctx.entries[1].text == "b2\n\n"


def test_refresh_specific_path_rereads_only_that_one(make_file):
    f1 = make_file("a.txt", "a1")
    f2 = make_file("b.txt", "b1")
    ctx = Context()
    ctx.add_file(path=f1)
    ctx.add_file(path=f2)
    f1.write_text("a2")
    f2.write_text("b2")
    count = ctx.refresh(path=f1)
    assert count == 1
    assert ctx.entries[0].text == "a2\n\n"
    assert ctx.entries[1].text == "b1\n\n"


def test_refresh_skips_text_only_entries(make_file):
    f = make_file("a.txt", "file body")
    ctx = Context()
    ctx.add(text="just text")
    ctx.add_file(path=f)
    f.write_text("updated")
    count = ctx.refresh()
    assert count == 1
    assert ctx.entries[0].text == "just text\n\n"
    assert ctx.entries[1].text == "updated\n\n"


def test_refresh_returns_zero_when_no_files():
    ctx = Context()
    ctx.add(text="just text")
    assert ctx.refresh() == 0


def test_refresh_raises_on_deleted_file(make_file):
    f = make_file("a.txt", "body")
    ctx = Context()
    ctx.add_file(path=f)
    f.unlink()
    with pytest.raises(FileNotFoundError):
        ctx.refresh()


# ============================================================
# add_summary()
# ============================================================

def test_add_summary_appends_provider_completion(make_file):
    f = make_file("model.py", "class Net: pass")
    p = MockProvider(completion_responses=["a focused summary"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f])
    assert len(ctx) == 1
    assert ctx.entries[0].text == "a focused summary\n\n"


def test_add_summary_uses_default_system_prompt(make_file):
    f = make_file("model.py", "class Net: pass")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f])
    assert p.calls[0]["system"] == defaults.SUMMARIZE_SYSTEM


def test_add_summary_user_message_includes_file_contents_and_name(make_file):
    f = make_file("model.py", "FILE_BODY_MARKER")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f])
    user = p.calls[0]["user"]
    assert "FILE_BODY_MARKER" in user
    assert "--- model.py ---" in user


def test_add_summary_concatenates_multiple_files_in_order(make_file):
    f1 = make_file("a.py", "FIRST")
    f2 = make_file("b.py", "SECOND")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f1, f2])
    user = p.calls[0]["user"]
    assert user.index("FIRST") < user.index("SECOND")


def test_add_summary_includes_hparam_names(make_file):
    f = make_file("model.py", "x = 1")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f], hparam_names=["lr", "depth"])
    user = p.calls[0]["user"]
    assert "lr" in user
    assert "depth" in user


def test_add_summary_omits_hparam_section_when_not_given(make_file):
    f = make_file("model.py", "x = 1")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f])
    assert "Hyperparameters being tuned" not in p.calls[0]["user"]


def test_add_summary_default_max_tokens_is_1000(make_file):
    f = make_file("x.py", "x")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f])
    assert p.calls[0]["max_tokens"] == 1000


def test_add_summary_max_tokens_override(make_file):
    f = make_file("x.py", "x")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f], max_tokens=200)
    assert p.calls[0]["max_tokens"] == 200


def test_add_summary_custom_system_prompt(make_file):
    f = make_file("x.py", "x")
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    custom = "Summarize in haiku form."
    ctx.add_summary(provider=p, paths=[f], system_prompt=custom)
    assert p.calls[0]["system"] == custom


def test_add_summary_entry_has_no_path_so_refresh_skips_it(make_file):
    """Summary entries are not file-backed — refresh() must not touch them."""
    f = make_file("x.py", "x")
    p = MockProvider(completion_responses=["the summary"])
    ctx = Context()
    ctx.add_summary(provider=p, paths=[f])
    assert ctx.entries[0].path is None
    assert ctx.refresh() == 0


def test_add_summary_raises_on_empty_paths():
    """Don't waste an LLM call on zero files — raise loudly instead."""
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    with pytest.raises(ValueError, match="at least one file"):
        ctx.add_summary(provider=p, paths=[])
    assert p.calls == []   # no LLM call happened


def test_add_summary_raises_on_missing_file(tmp_path):
    p = MockProvider(completion_responses=["s"])
    ctx = Context()
    with pytest.raises(FileNotFoundError):
        ctx.add_summary(provider=p, paths=[tmp_path / "missing.py"])


# ============================================================
# pop() / clear()
# ============================================================

def test_pop_default_removes_last():
    ctx = Context()
    ctx.add(text="a")
    ctx.add(text="b")
    ctx.add(text="c")
    removed = ctx.pop()
    assert removed.text == "c\n\n"
    assert [e.text for e in ctx.entries] == ["a\n\n", "b\n\n"]


def test_pop_index_removes_that_one():
    ctx = Context()
    ctx.add(text="a")
    ctx.add(text="b")
    ctx.add(text="c")
    removed = ctx.pop(index=0)
    assert removed.text == "a\n\n"
    assert [e.text for e in ctx.entries] == ["b\n\n", "c\n\n"]


def test_pop_negative_index():
    ctx = Context()
    ctx.add(text="a")
    ctx.add(text="b")
    ctx.add(text="c")
    removed = ctx.pop(index=-2)
    assert removed.text == "b\n\n"


def test_pop_on_empty_raises():
    with pytest.raises(IndexError):
        Context().pop()


def test_pop_out_of_range_raises():
    ctx = Context()
    ctx.add(text="a")
    with pytest.raises(IndexError):
        ctx.pop(index=5)


def test_clear_removes_all():
    ctx = Context()
    ctx.add(text="a")
    ctx.add(text="b")
    ctx.clear()
    assert len(ctx) == 0


def test_clear_on_empty_is_noop():
    ctx = Context()
    ctx.clear()
    assert len(ctx) == 0


# ============================================================
# render()
# ============================================================

def test_render_empty_context():
    assert Context().render() == ""


def test_render_one_entry():
    ctx = Context()
    ctx.add(text="hello")
    assert ctx.render() == "hello\n\n"


def test_render_concatenates_entries_in_order():
    ctx = Context()
    ctx.add(text="A")
    ctx.add(text="B")
    ctx.add(text="C")
    assert ctx.render() == "A\n\nB\n\nC\n\n"


def test_render_equals_sum_of_entry_texts():
    """The 'exactly what the LLM sees' contract: render() output equals
    the plain concatenation of every Entry.text — no separator added at
    render time."""
    ctx = Context()
    ctx.add(text="alpha")
    ctx.add(text="beta")
    assert ctx.render() == ctx.entries[0].text + ctx.entries[1].text


# ============================================================
# __len__
# ============================================================

def test_len_empty():
    assert len(Context()) == 0


def test_len_after_adds():
    ctx = Context()
    ctx.add(text="a")
    ctx.add(text="b")
    ctx.add(text="c")
    assert len(ctx) == 3


# ============================================================
# to_dict / from_dict round-trip
# ============================================================

def test_to_dict_empty_context():
    assert Context().to_dict() == {"entries": []}


def test_to_dict_text_only_entry():
    ctx = Context()
    ctx.add(text="hello")
    assert ctx.to_dict() == {"entries": [{"text": "hello\n\n", "path": None}]}


def test_to_dict_file_entry_serializes_path_as_string(make_file):
    f = make_file("note.txt", "body")
    ctx = Context()
    ctx.add_file(path=f)
    d = ctx.to_dict()
    assert d["entries"][0]["text"] == "body\n\n"
    assert d["entries"][0]["path"] == str(f)
    assert isinstance(d["entries"][0]["path"], str)


def test_to_dict_is_json_serializable(make_file):
    """to_dict() output must round-trip through json.dumps without a custom encoder."""
    f = make_file("note.txt", "body")
    ctx = Context()
    ctx.add(text="text only")
    ctx.add_file(path=f)
    json.dumps(ctx.to_dict())


def test_from_dict_reconstructs_text_only_entry():
    original = Context()
    original.add(text="hello")
    restored = Context.from_dict(data=original.to_dict())
    assert len(restored) == 1
    assert restored.entries[0].text == "hello\n\n"
    assert restored.entries[0].path is None


def test_from_dict_reconstructs_file_entry(make_file):
    f = make_file("note.txt", "body")
    original = Context()
    original.add_file(path=f)
    restored = Context.from_dict(data=original.to_dict())
    assert restored.entries[0].text == "body\n\n"
    assert restored.entries[0].path == f


def test_round_trip_preserves_render(make_file):
    f = make_file("note.txt", "body")
    original = Context()
    original.add(text="alpha")
    original.add_file(path=f)
    original.add(text="beta")
    restored = Context.from_dict(data=original.to_dict())
    assert restored.render() == original.render()
