"""Reviewing a result: opening the interface on it, or reading it in the terminal.

`confoundsearch review --in FILE`, and `run --review` for the page it just wrote, take a format. `ui`, the
default, opens the page in a browser: the diagram of confounds with the papers behind each link. `text` prints
the same confounds, each with the papers for both links and their quotations, as plain text.
"""

from __future__ import annotations

import sys
import webbrowser
import zipfile
from pathlib import Path
from typing import Any, Callable

from confoundsearch.interface.ui.page import byline, papers, yes

FORMATS = ("ui", "text")
QUOTE_LIMIT = 300


def review_text(data: dict[str, Any]) -> str:
    """The confounds in a `share_data` dict as plain text: each with its papers for both links."""
    iv, dv = data["iv"], data["dv"]
    by_name = {c["name"]: c for c in data.get("dag_confounders", [])}
    names = [n for n in data.get("all_md_names", []) if n in by_name]
    lines = [f"Candidate confounds for {iv} → {dv}", ""]
    if not names:
        lines.append("No confounds found.")
    for number, name in enumerate(names, 1):
        confound = by_name[name]
        to_iv, to_dv = yes(confound["evidence_to_iv"]), yes(confound["evidence_to_dv"])
        lines.append(f"{number}. {name}  ({papers(len(to_iv))} for {iv}, {papers(len(to_dv))} for {dv})")
        for target, entries in ((iv, to_iv), (dv, to_dv)):
            lines.append(f"   {name} affects {target}:")
            lines += [line for entry in entries for line in entry_lines(entry)] or ["     (no papers)"]
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def entry_lines(entry: dict[str, Any]) -> list[str]:
    title = entry.get("title") or "Unknown"
    head = f"     - {title}" + (f" ({byline(entry)})" if byline(entry) else "")
    out = [head]
    if entry.get("url"):
        out.append(f"       {entry['url']}")
    quote = (entry.get("quote") or "").strip()
    if quote:
        quote = quote if len(quote) <= QUOTE_LIMIT else quote[:QUOTE_LIMIT].rstrip() + "…"
        out.append(f"       “{quote}”")
    return out


def load_share(path: Path) -> dict[str, Any] | None:
    """A `share_data` dict from a `.zip` or `.json` file, or None for a file that is already a page."""
    from confoundsearch.interface.known import known_to_share_data, load_known

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            import json

            data: dict[str, Any] = json.loads(archive.read("share_data.json"))
            return data
    if path.suffix in (".html", ".htm"):
        return None
    data = load_known(path)
    return known_to_share_data(data) if "controls" in data else data


def open_page(
    page: Path, opener: Callable[[str], bool] = webbrowser.open, err: Callable[[str], None] | None = None
) -> bool:
    """Open `page` in the browser. With no browser (a remote session), say where the file is instead."""
    say = err or (lambda message: print(message, file=sys.stderr))
    opened = bool(opener(page.resolve().as_uri()))
    if not opened:
        say(f"could not open a browser; open this file yourself: {page.resolve()}")
    return opened


def where_it_went(
    page: Path, audit: Path | None, intact: bool | None, log: Path | None = None, review_hint: bool = True
) -> str:
    """The closing message of a run: exactly where the files are and how to look at them."""
    page = page.resolve()
    lines = ["", f"Results are in {page.parent}", f"  page   {page}   (the interface: open it in a browser)"]
    if audit is not None:
        state = "" if intact is None else f", hash chain {'intact' if intact else 'BROKEN'}"
        lines.append(f"  audit  {audit.resolve()}   (every query and verdict{state})")
    if log is not None:
        lines.append(
            f"  log    {log.resolve()}   (every search request and step: read this first if a run stops)"
        )
    if review_hint:
        lines += ["", f"To review: confoundsearch review --in {page}"]
    return "\n".join(lines)
