"""Context — ordered text log fed to the LLM on every suggest().

The Context is a first-class object in llmtuna: it accumulates every piece
of text the LLM should see — scientist notes, file snapshots, prior trial
results, raw LLM outputs — and is rendered into the user message on each
Tuner.suggest() call. It survives save/load cycles via to_dict/from_dict.

The ``add_summary`` method runs a one-shot LLM call that digests source
files into a focused prompt — useful when raw file snapshots would blow
up the context window on long runs.
"""

from dataclasses import dataclass
from pathlib import Path

from llmtuna import defaults
from llmtuna.providers.base import Provider

_ENTRY_SUFFIX = "\n\n"


@dataclass
class Entry:
    """A single ordered item in the context log.

    Attributes:
        text: Exactly what the LLM sees for this entry. Includes any
            trailing separator that ``add()`` / ``add_file()`` appended
            on insertion, so ``Context.render()`` is plain concatenation.
        path: If this entry came from ``add_file()``, the source path on
            disk. Used by ``refresh()`` to re-read the file. None for
            free-form text entries.
    """

    text: str
    path: Path | None = None


class Context:
    """Ordered, mutable text log fed to the LLM.

    The Context is the canonical "what does the LLM know" object. Add
    arbitrary text via ``add()``, snapshot files via ``add_file()``,
    re-read snapshotted files via ``refresh()``, and prune via
    ``pop()`` / ``clear()``. The full content is serialized into the
    user message on each ``Tuner.suggest()`` call.

    Internally this is just a list of Entry records. There is no taxonomy
    of entry kinds — anything destined for the LLM is plain text. The
    ordering of entries is the only structure. Each entry is stored with
    its trailing separator already in ``Entry.text``, so ``render()`` is
    plain concatenation and ``Entry.text`` is exactly what the LLM sees
    for that entry.
    """

    def __init__(self):
        """Create an empty context with no entries."""
        self.entries: list[Entry] = []

    def add(self, text: str) -> None:
        """Append a free-form text entry to the log.

        A standard separator is appended to the stored text so that
        ``Entry.text`` is exactly what the LLM sees for this entry, and
        consecutive entries in ``render()`` are visually separated.

        Args:
            text: Any string. Stored verbatim with a trailing separator.
        """
        self.entries.append(Entry(text=text + _ENTRY_SUFFIX))

    def add_summary(
        self,
        provider: Provider,
        paths: list[str | Path],
        *,
        hparam_names: list[str] | None = None,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
    ) -> None:
        """Read source files, have the LLM summarize them, append the summary.

        Use this instead of multiple ``add_file()`` calls when raw source
        would blow up the context window on long runs. Costs one extra
        LLM call upfront in exchange for much smaller per-trial context.

        The resulting entry is a normal text entry — it has no ``path``
        backing and is therefore NOT updated by ``refresh()``. To
        re-summarize, ``pop()`` the old summary and call
        ``add_summary()`` again.

        Args:
            provider: The ``Provider`` whose ``complete()`` method makes
                the actual summarization call. Often the same provider
                you give to the ``Tuner``.
            paths: Source files to read and summarize. Snapshotted at
                call time.
            hparam_names: Optional list of hyperparameter names to focus
                the summary on. Strongly recommended.
            max_tokens: Cap on the LLM's summary output. Default 1000
                tokens (~700 words) matches the prompt's 300-500 word
                target with headroom.
            system_prompt: Optional override for the default
                ``SUMMARIZE_SYSTEM`` prompt.

        Raises:
            ValueError: If ``paths`` is empty (no point making an LLM
                call to summarize nothing).
            FileNotFoundError: If any of ``paths`` does not exist.
        """
        if not paths:
            raise ValueError(
                "Context.add_summary: paths must contain at least one file"
            )
        parts: list[str] = []
        for p in paths:
            path = Path(p)
            parts.append(f"--- {path.name} ---\n{path.read_text()}")
        source_text = "\n\n".join(parts)

        hparam_info = ""
        if hparam_names:
            hparam_info = (
                f"\n\nHyperparameters being tuned: {', '.join(hparam_names)}."
            )

        user_msg = (
            "Summarize the following source files for the purpose of "
            f"hyperparameter optimization.{hparam_info}\n\n{source_text}"
        )

        summary = provider.complete(
            system=system_prompt or defaults.SUMMARIZE_SYSTEM,
            user=user_msg,
            max_tokens=max_tokens,
        )
        self.add(text=summary)

    def add_file(self, path: str | Path) -> None:
        """Read a file at this moment (snapshot) and append its contents.

        The file is read at call time and the contents stored in the entry,
        with a trailing separator appended. Subsequent edits to the file on
        disk do not affect the entry unless ``refresh()`` is called.

        Args:
            path: Path to the file. Accepts ``str`` or ``pathlib.Path``.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        text = path.read_text() + _ENTRY_SUFFIX
        self.entries.append(Entry(text=text, path=path))

    def refresh(self, path: str | Path | None = None) -> int:
        """Re-read file-backed entries from disk and update their text in place.

        Use this when a snapshotted file has changed and the LLM should see
        the new contents on the next ``suggest()``. The trailing separator
        is re-applied so ``Entry.text`` remains exactly what the LLM will
        see.

        Args:
            path: If provided, refresh only entries that originally came
                from this path. If None, refresh every file-backed entry.

        Returns:
            The number of entries that were re-read.

        Raises:
            FileNotFoundError: If a tracked path no longer exists on disk.
                Refresh is partial: entries updated before the failure
                remain updated; entries after are left at their previous
                text.
        """
        target = Path(path) if path is not None else None
        count = 0
        for entry in self.entries:
            if entry.path is None:
                continue
            if target is not None and entry.path != target:
                continue
            entry.text = entry.path.read_text() + _ENTRY_SUFFIX
            count += 1
        return count

    def pop(self, index: int = -1) -> Entry:
        """Remove and return the entry at the given position.

        Use this to prune entries you no longer want the LLM to see — a
        trial result you've decided was buggy, stale advice that's
        confusing the search, an oversized file snapshot, etc.

        Args:
            index: Position to remove. Negative indexes count from the
                end. Default ``-1`` removes the most recent entry.

        Returns:
            The removed ``Entry``.

        Raises:
            IndexError: If the context is empty or the index is out of
                range.
        """
        return self.entries.pop(index)

    def clear(self) -> None:
        """Remove every entry from the context.

        Useful when reusing the same ``Tuner`` across distinct phases of
        a run and you want a clean slate for the LLM.
        """
        self.entries.clear()

    def render(self) -> str:
        """Concatenate all entries into the single text block sent to the LLM.

        This is plain string concatenation — each entry's text already
        carries its trailing separator from ``add()`` / ``add_file()``.
        The returned string is exactly what the LLM sees as the rendered
        context.

        Returns:
            The full rendered context as a string. Empty string if there
            are no entries.
        """
        return "".join(e.text for e in self.entries)

    def __len__(self) -> int:
        """Return the number of entries currently in the context."""
        return len(self.entries)

    def to_dict(self) -> dict:
        """Serialize the context to a JSON-safe dict for ``Tuner.save()``.

        Returns:
            A dict of the form
            ``{"entries": [{"text": ..., "path": ...}, ...]}``. The
            ``path`` value is a string or None.
        """
        return {
            "entries": [
                {
                    "text": e.text,
                    "path": str(e.path) if e.path is not None else None,
                }
                for e in self.entries
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """Reconstruct a Context from ``to_dict()`` output.

        Args:
            data: A dict produced by ``to_dict()``. Must contain an
                ``"entries"`` key whose value is a list of entry dicts.

        Returns:
            A new Context with the same entries (text, path).
        """
        ctx = cls()
        for d in data["entries"]:
            ctx.entries.append(
                Entry(
                    text=d["text"],
                    path=Path(d["path"]) if d.get("path") is not None else None,
                )
            )
        return ctx
