"""The result of a run as plain data, for programs and agents to read instead of scraping text.

    confoundsearch run --json ...        prints this to standard output

Every passage named here is one the run really retrieved and the judge accepted; the full text, the query
that found it and the step it was judged at are in the audit trail whose path is under "files".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from confoundsearch.logic.audit import EvidenceRow
from confoundsearch.logic.episode import Trajectory

QUOTE_CHARS = 400


def _passage(row: EvidenceRow) -> dict[str, Any]:
    quote = row.text if len(row.text) <= QUOTE_CHARS else row.text[:QUOTE_CHARS].rstrip() + "..."
    return {
        "id": row.passage_id,
        "title": row.title,
        "url": row.url,
        "quote": quote,
        "found_by_query": row.retrieved_by,
        "judged_at_step": row.judged_at_step,
    }


def run_result(trajectory: Trajectory, page: Path | None = None, audit: Path | None = None) -> dict[str, Any]:
    """Confounds with their supporting passages, the score, how the run ended, and the audit status."""
    trail = trajectory.audit_trail()
    summary = trail.to_dict()
    graph = trajectory.final.graph
    ranked: list[tuple[bool, int, str, dict[str, Any]]] = []
    for vertex in graph.confounds:
        label = vertex.label
        to_iv, to_dv = trail.supporting(label, "X"), trail.supporting(label, "Y")
        rejected = [r for r in trail.evidence if r.confound == label and r.verdict == "rejected"]
        entry = {
            "label": label,
            "phrasings": sorted(vertex.strings),
            "linked_to_both_variables": graph.is_connected(label),
            "supported_on_both_links": bool(to_iv and to_dv),
            "evidence_to_iv": [_passage(r) for r in to_iv],
            "evidence_to_dv": [_passage(r) for r in to_dv],
            "rejected_passages": len(rejected),
        }
        ranked.append((not (to_iv and to_dv), -(len(to_iv) + len(to_dv)), label, entry))
    confounds = [entry for *_, entry in sorted(ranked, key=lambda r: r[:3])]
    return {
        "question": summary["question"],
        "ended": summary["ended"],
        "steps": len(trajectory.steps),
        "score": trajectory.score,
        "confounds": confounds,
        "model_calls": summary["model_calls"],
        "models": summary["models"],
        "files": {"page": str(page) if page else None, "audit": str(audit) if audit else None},
        "audit": {k: summary[k] for k in ("events", "intact", "complete", "head")},
    }
