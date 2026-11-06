"""State -> the `share_data.json` format read by the ConfoundSearch interface.

The interface (the Share tab's standalone HTML, see `confoundsearch.interface.ui`) draws
meta-dimensions containing confounders, each with evidence for its two legs.
Here every graph vertex becomes one meta-dimension holding one confounder of
the same name, and the vertex's strings are listed in the meta-dimension's description.
Attached passages are labelled "Yes" and judge-rejected ones "No".

Names that reach the page are stripped of angle brackets and quotes, because
the interface interpolates them into HTML.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from confoundsearch.logic.audit import AuditLog
from confoundsearch.logic.graph import TARGETS, Passage, Target
from confoundsearch.logic.reward import phi
from confoundsearch.logic.state import State

SCHEMA = "share_data/1"


def plain(text: str) -> str:
    """Make `text` safe to drop into HTML without escaping."""
    return text.translate({ord("<"): None, ord(">"): None, ord('"'): "\u201d", ord("'"): "\u2019"})


def _entry(passage: Passage, label: str, highlight: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {
        "title": plain(passage.title or passage.text[:80]),
        "url": passage.url if passage.url.startswith(("http://", "https://")) else "",
        "quote": passage.text,
        "llm_label": label,
    }
    for field in ("authors", "venue"):  # shown under the title when the source gave them
        if getattr(passage, field):
            entry[field] = plain(getattr(passage, field))
    for field in ("year", "cited_by"):
        if getattr(passage, field) is not None:
            entry[field] = getattr(passage, field)
    if highlight:
        entry["highlight"] = highlight  # the sentence of `quote` that best states the claim
    return entry


def most_cited_first(passage: Passage) -> tuple[int, str, str]:
    """A sort key: the more a paper is cited the sooner it comes, then by title; papers with no count last."""
    return (-(passage.cited_by if passage.cited_by is not None else -1), passage.title.casefold(), passage.id)


def choose_highlights(
    state: State, highlight: Callable[[str, str, Sequence[Passage]], Mapping[str, str]]
) -> dict[tuple[str, Target, str], str]:
    """Ask `highlight` once for each link that has papers: which sentence of each best states the claim.

    Returns {(confound, target, passage id): sentence}. A link it cannot answer for simply has none.
    """
    chosen: dict[tuple[str, Target, str], str] = {}
    for vertex in state.graph.confounds:
        for target in TARGETS:
            cited = sorted(state.graph.passages(vertex.label, target), key=most_cited_first)
            if not cited:
                continue
            for passage_id, sentence in highlight(
                vertex.label, state.graph.target_name(target), cited
            ).items():
                chosen[(vertex.label, target, passage_id)] = sentence
    return chosen


def _evidence(
    state: State, label: str, target: Target, highlights: Mapping[tuple[str, Target, str], str]
) -> list[dict[str, Any]]:
    entries = [
        _entry(p, "Yes", highlights.get((label, target, p.id), ""))
        for p in sorted(state.graph.passages(label, target), key=most_cited_first)
    ]
    rejected = {pid for (lbl, tgt), pid in state.rejected if (lbl, tgt) == (label, target)}
    entries += [_entry(p, "No") for p in sorted(state.observed, key=lambda p: p.id) if p.id in rejected]
    return entries


def to_share_data(
    state: State, highlights: Mapping[tuple[str, Target, str], str] | None = None
) -> dict[str, Any]:
    """The result for the interface. Papers come most-cited first; `highlights` maps (confound, target,
    passage id) to the sentence of that paper which best states the link."""
    graph = state.graph
    marked = highlights or {}
    confounders, meta_dimensions, names = [], [], []
    for vertex in graph.confounds:
        label = plain(vertex.label)
        strings = ", ".join(sorted(plain(s) for s in vertex.strings))
        evidence = {t: _evidence(state, vertex.label, t, marked) for t in TARGETS}
        confounders.append({"name": label, "evidence_to_iv": evidence["X"], "evidence_to_dv": evidence["Y"]})
        meta_dimensions.append(
            {
                "name": label,
                "description": f"Cluster of strings: {strings}.",
                "groups": [{"name": "Evidence", "description": "", "confounders": [label]}],
            }
        )
        if all(any(e["llm_label"] == "Yes" for e in evidence[t]) for t in TARGETS):
            names.append(label)
    return {
        "schema": SCHEMA,
        "mode": "explore",
        "iv": plain(graph.x),
        "dv": plain(graph.y),
        "dag_confounders": confounders,
        "dag_grouped": {"meta_dimensions": meta_dimensions},
        "all_md_names": names,
        "phi": phi(graph),
        "done": state.done,
    }


def export_zip(data: dict[str, Any], audit: AuditLog | None = None) -> bytes:
    """`data` zipped as share_data.json, the format the interface's Share tab accepts.

    With `audit`, the run's audit trail travels in the same file as audit.jsonl.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("share_data.json", json.dumps(data, ensure_ascii=False))
        if audit is not None:
            archive.writestr("audit.jsonl", audit.to_jsonl())
    return buffer.getvalue()


def save_export(state: State, path: str | Path, audit: AuditLog | None = None) -> Path:
    """Write the result of a run to a file for the interface to display: `.json` or `.zip`.

    Pass the run's `audit` log to put its audit trail in the `.zip` too.
    """
    target = Path(path)
    data = to_share_data(state)
    if audit is not None and target.suffix != ".zip":
        raise ValueError("an audit trail goes in a .zip export; save it separately with audit.save(...)")
    if target.suffix == ".json":
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif target.suffix == ".zip":
        target.write_bytes(export_zip(data, audit))
    else:
        raise ValueError(f"export to .json or .zip, not {target.suffix or 'a file with no extension'}")
    return target
