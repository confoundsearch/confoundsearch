"""Papers kept on this machine in small databases, so a search is only ever made once.

Every search a run makes is saved, keyed by the source and the query terms, and the same search later is
answered from disk: no request, no rate limit, and the same papers as last time. It fills up as you retrieve.
Like Hugging Face's model folder, this data directory is set by a variable and has a default:

    CONFOUNDSEARCH_CACHE=~/confoundsearch-data     # anywhere you like, to keep what is downloaded
    <temp folder>/confoundsearch-<user>            # the default: /tmp on Linux, which the system clears
                                                   # from time to time (so nothing here is precious)

There is one SQLite file per question, that is per pair of independent and dependent variable, so a new
pair starts a new database and nothing leaks between studies. Inside, `searches` holds each search (source,
query terms, how many) and the ranked list of papers it returned; `papers` holds each paper once, whichever
source or search found it: papers are the same when their titles match (case and punctuation aside), and
when two copies differ the one with the fuller abstract is kept, so a source that returned no abstract is
filled in by another.

SQLite lets one writer in at a time, so reads happen where they are asked but every write goes into a queue
that a single background thread empties, in order. `wait_for_writes()` returns when the queue is empty.

    CachedCorpus(SemanticScholarCorpus(key), cache_dir(), "semanticscholar", scope=lambda: (iv, dv))
    flush(cache_dir())                         # empty it: `confoundsearch flush`

Only papers and the questions asked are stored, never keys. A search that failed is not stored.
"""

from __future__ import annotations

import atexit
import contextlib
import getpass
import hashlib
import json
import os
import queue
import re
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from confoundsearch.data.corpus import Corpus
from confoundsearch.data.multi import title_key
from confoundsearch.logic.graph import Passage

ENV_VAR = "CONFOUNDSEARCH_CACHE"
PREFIX, SUFFIX = "papers-", ".sqlite3"  # the only files this module makes (plus SQLite's own beside them)
SIDE_FILES = ("", "-journal", "-wal", "-shm")
SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    key TEXT PRIMARY KEY, id TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL,
    text TEXT NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS searches (
    source TEXT NOT NULL, query TEXT NOT NULL, k INTEGER NOT NULL, papers TEXT NOT NULL, saved TEXT NOT NULL,
    PRIMARY KEY (source, query, k)
);
"""


def default_dir() -> Path:
    """The system's temporary folder, in a subfolder of its own for this user (/tmp is shared on Linux)."""
    try:
        user = getpass.getuser()
    except (KeyError, OSError):  # no user name to be found (some containers)
        user = "user"
    return Path(tempfile.gettempdir()) / f"confoundsearch-{re.sub(r'[^A-Za-z0-9_.-]', '_', user)}"


def cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Where papers are kept: $CONFOUNDSEARCH_CACHE, else a folder in the system's temporary directory."""
    settings: Mapping[str, str] = os.environ if environ is None else environ
    if settings.get(ENV_VAR):
        return Path(settings[ENV_VAR]).expanduser()
    return default_dir()


def paper_key(passage: Passage, source: str) -> str:
    """What makes two copies the same paper: its title; with no title, only the same source and id."""
    return title_key(passage.title) or f"{source}:{passage.id}"


def _plain(text: str) -> str:
    return " ".join(text.casefold().split())


def database_path(folder: Path, question: tuple[str, str] | None) -> Path:
    """The file for one pair of variables: readable, safe as a file name, different for each pair."""
    if question is None:
        return Path(folder) / f"{PREFIX}unscoped{SUFFIX}"  # a search made outside any run
    iv, dv = _plain(question[0]), _plain(question[1])
    words = [re.sub(r"[^a-z0-9]+", "-", part).strip("-")[:30] or "x" for part in (iv, dv)]
    tag = hashlib.sha1(f"{iv}\0{dv}".encode()).hexdigest()[:8]  # noqa: S324 (a name, not security)
    return Path(folder) / f"{PREFIX}{words[0]}--{words[1]}-{tag}{SUFFIX}"


class WriteQueue:
    """Every write goes through here: one thread, one job at a time, in the order they were asked."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, job: Callable[[], None]) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, daemon=True)
                self._thread.start()
        self._jobs.put(job)

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception:  # noqa: BLE001  (saving is a courtesy: a failed write must not end the thread)
                pass
            finally:
                self._jobs.task_done()

    def wait(self) -> None:
        """Return once every write submitted so far has been made."""
        self._jobs.join()


WRITES = WriteQueue()
atexit.register(WRITES.wait)  # a run that ends with writes waiting still saves them


def wait_for_writes() -> None:
    WRITES.wait()


class PaperDatabase:
    """The searches and papers saved in one file. A connection per call, so threads can share it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)  # /tmp is shared: keep it private
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with contextlib.suppress(sqlite3.Error):  # readers then never wait for the writer
                connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _asked(query: str) -> str:
        return _plain(query)  # case and spacing do not change what is asked

    def load(self, source: str, query: str, k: int) -> list[Passage] | None:
        """The papers this search found last time, best first, or None if it was never made."""
        if not self.path.exists():
            return None
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT papers FROM searches WHERE source = ? AND query = ? AND k = ?",
                    (source, self._asked(query), k),
                ).fetchone()
                if row is None:
                    return None
                found = []
                for key in json.loads(row[0]):
                    paper = db.execute(
                        "SELECT id, text, title, url FROM papers WHERE key = ?", (key,)
                    ).fetchone()
                    if paper is None:  # the table was changed by hand: ask again
                        return None
                    found.append(Passage(*paper))
                return found
        except (sqlite3.Error, ValueError, TypeError, OSError):  # damaged: ask again
            return None

    def save(self, source: str, query: str, k: int, found: list[Passage]) -> None:
        """Remember the answer, and each paper once: the copy with the fuller abstract wins."""
        try:
            with self._connect() as db:
                keys = []
                for passage in found:
                    key = paper_key(passage, source)
                    keys.append(key)
                    row = db.execute("SELECT text, url, title FROM papers WHERE key = ?", (key,)).fetchone()
                    if row is None:
                        db.execute(
                            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                            (key, passage.id, passage.title, passage.url, passage.text, source),
                        )
                    else:
                        text, url, title = row
                        db.execute(
                            "UPDATE papers SET text = ?, url = ?, title = ? WHERE key = ?",
                            (
                                passage.text if len(passage.text) > len(text) else text,
                                url or passage.url,
                                title or passage.title,
                                key,
                            ),
                        )
                db.execute(
                    "INSERT OR REPLACE INTO searches VALUES (?, ?, ?, ?, ?)",
                    (
                        source,
                        self._asked(query),
                        k,
                        json.dumps(keys),
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ),
                )
        except (sqlite3.Error, OSError):  # a read-only or full disk must not stop a search that worked
            pass

    def counts(self) -> tuple[int, int]:
        """(searches, papers) saved."""
        try:
            with self._connect() as db:
                return (
                    db.execute("SELECT COUNT(*) FROM searches").fetchone()[0],
                    db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
                )
        except (sqlite3.Error, OSError):
            return 0, 0


class CachedCorpus:
    """`corpus`, remembering each answer under the name of its source, in the database of the question being
    asked. `scope()` says which question that is (None outside a run); writes go through the write queue."""

    def __init__(
        self,
        corpus: Corpus,
        folder: Path,
        name: str,
        log: Callable[..., None] | None = None,
        scope: Callable[[], tuple[str, str] | None] | None = None,
    ) -> None:
        self.corpus, self.folder, self.name, self.log, self.scope = corpus, Path(folder), name, log, scope

    def database(self) -> PaperDatabase:
        return PaperDatabase(database_path(self.folder, self.scope() if self.scope else None))

    def search(self, query: str, k: int) -> list[Passage]:
        database = self.database()
        saved = database.load(self.name, query, k)
        if saved is not None:
            if self.log:
                self.log("cache_hit", source=self.name, query=query, papers=len(saved))
            return saved
        found = self.corpus.search(query, k)  # a failure raises here and is not stored
        WRITES.submit(lambda: database.save(self.name, query, k, found))
        if self.log:
            self.log("cache_miss", source=self.name, query=query, papers=len(found))
        return found


def databases(folder: Path, question: tuple[str, str] | None = None) -> list[Path]:
    """The database files in `folder`: every question's, or just this one's."""
    if question is not None:
        path = database_path(folder, question)
        return [path] if path.is_file() else []
    root = Path(folder)
    return sorted(root.glob(f"{PREFIX}*{SUFFIX}")) if root.is_dir() else []


def flush(folder: Path, question: tuple[str, str] | None = None) -> tuple[int, int, int]:
    """Delete the saved searches under `folder`, or only one question's. Returns (searches, papers, bytes).

    Only the database files and SQLite's own leftovers beside them are removed; anything else in a folder you
    pointed the variable at stays, and the folder itself is removed only if that leaves it empty.
    """
    WRITES.wait()  # let anything still being saved land first, so it is not left behind
    searches = papers = freed = 0
    for path in databases(folder, question):
        found = PaperDatabase(path).counts()
        searches, papers = searches + found[0], papers + found[1]
        for side in SIDE_FILES:
            extra = path.with_name(path.name + side)
            if extra.is_file():
                freed += extra.stat().st_size
                extra.unlink()
    with contextlib.suppress(OSError):
        Path(folder).rmdir()  # only if empty
    return searches, papers, freed
