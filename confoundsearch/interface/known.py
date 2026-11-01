"""Known confounds -> the `share_data.json` format, so the interface can show them.

The input is what an expert already knows: for X and Y, a list of candidate controls, each with the papers (a
title, a DOI and a quotation) that support its link to each variable. `known_to_share_data` turns it into the
same format `confoundsearch.interface.export` writes for a run, so one page shows both an agent's finding and
the expert's answer. Every paper is labelled "Yes": an expert, not the AI judge, attached it.

    {"iv": "...", "dv": "...", "controls": [
        {"name": "...", "citations_to_IV": [{"title", "doi", "quotation"}], "citations_to_DV": [...]}]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from confoundsearch.interface.export import SCHEMA, plain
from confoundsearch.interface.ui.page import papers, standalone_page


def _entry(paper: dict[str, str]) -> dict[str, str]:
    doi = paper.get("doi", "")
    entry = {
        "title": plain(paper["title"]),
        "url": f"https://doi.org/{doi}" if doi else "",
        "quote": paper["quotation"],
        "llm_label": "Yes",
    }
    for field in ("authors", "venue", "year"):  # shown under the title when the file gives them
        if paper.get(field):
            entry[field] = plain(str(paper[field]))
    return entry


def known_to_share_data(known: dict[str, Any]) -> dict[str, Any]:
    confounders, meta_dimensions, names = [], [], []
    for control in known["controls"]:
        label = plain(control["name"])
        to_iv = [_entry(p) for p in control.get("citations_to_IV", [])]
        to_dv = [_entry(p) for p in control.get("citations_to_DV", [])]
        confounders.append({"name": label, "evidence_to_iv": to_iv, "evidence_to_dv": to_dv})
        meta_dimensions.append(
            {
                "name": label,
                "description": (
                    f"{papers(len(to_iv))} to {known['iv']} (IV), {papers(len(to_dv))} to {known['dv']} (DV)."
                ),
                "groups": [{"name": "Evidence", "description": "", "confounders": [label]}],
            }
        )
        names.append(label)  # an expert's list is shown whole, supported on both legs or not
    return {
        "schema": SCHEMA,
        "mode": "explore",
        "iv": plain(known["iv"]),
        "dv": plain(known["dv"]),
        "dag_confounders": confounders,
        "dag_grouped": {"meta_dimensions": meta_dimensions},
        "all_md_names": names,
    }


def load_known(path: str | Path) -> dict[str, Any]:
    known: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return known


def page_from_json(data: dict[str, Any]) -> str:
    """The standalone page for a JSON blob: known confounds, or a `share_data.json` a run exported."""
    if "controls" in data:
        data = known_to_share_data(data)
    elif "dag_confounders" not in data:
        raise ValueError('not a known-confounds file (no "controls") and not a share_data file')
    return standalone_page(data)
