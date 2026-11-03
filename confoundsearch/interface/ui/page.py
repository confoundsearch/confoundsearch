# ruff: noqa: E501  (long HTML strings)
"""The standalone page: the interface's one self-contained HTML file.

One card per confound, a diagram with the first few of them drawn between the two variables, and the papers behind
each link. Clicking an arrow (or a card's link) shows that edge and its papers; clicking a node or a card shows both
of the confound's edges. The selected edge is drawn strong and the rest fade; the evidence panel names the edge. `standalone_page(data)` takes the `share_data.json`
dict that `confoundsearch.interface.export` and `confoundsearch.interface.known` write. The styling (`page.css`)
and the click handling (`page.js`) sit next to this file.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

MAX_DRAWN = 5  # confounds drawn in the diagram; every confound still has its card
COLORS = (
    "#e66fc3",
    "#a8932c",
    "#0e9fb3",
    "#6d5bd0",
    "#d9822b",
    "#2f9e6a",
    "#c2410c",
    "#3b82f6",
    "#84a312",
    "#b8478a",
    "#0891b2",
    "#9a6b3f",
)  # fmt: skip  (one colour per confound: its node, its arrows, its cards and its evidence all use it)
QUOTE_LIMIT = 800  # a longer quotation is cut here, with an ellipsis
ARC_CENTER, ARC_RADIUS, ARC_SPAN = (
    (320.0, 388.4),
    258.4,
    45.72,
)  # nodes sit on an arc, 45.72 degrees to a side
VARIABLES = {"x": (170.0, 310.0), "y": (470.0, 310.0)}
VARIABLE_R, NODE_R = 35.0, 22.0


@lru_cache(maxsize=None)
def asset(name: str) -> str:
    return (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")


def papers(n: int) -> str:
    """'1 paper', '2 papers'."""
    return f"{n} paper" + ("" if n == 1 else "s")


def yes(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the papers the judge, or the expert, accepted."""
    return [e for e in entries if e.get("llm_label") == "Yes"]


def byline(entry: dict[str, Any]) -> str:
    parts = [str(entry.get("authors") or ""), str(entry.get("venue") or ""), str(entry.get("year") or "")]
    return " · ".join(p for p in parts if p)


def quotation(text: str) -> str:
    text = text if len(text) <= QUOTE_LIMIT else text[:QUOTE_LIMIT].rstrip() + "…"
    return f"“{html.escape(text)}”"


def passage(entry: dict[str, Any], first: bool, name: str, target: str) -> str:
    """One paper: which link it supports, the quoted passage as the main text, then its title and byline."""
    border = "margin-top:6px;" if first else "border-top:1px solid #e6e9ee;margin-top:6px;padding-top:12px;"
    title = html.escape(entry.get("title") or "Unknown")
    url = entry.get("url") or ""
    link = f'<a href="{html.escape(url)}" target="_blank">{title}</a>' if url else title
    lines = [
        f'<div class="passage-entry" style="{border}">',
        f'<div class="supports">Supports: {html.escape(name)} \u2192 {html.escape(target)}</div>',
    ]
    if entry.get("quote"):
        lines.append(f'<p class="quote">{quotation(entry["quote"])}</p>')
    lines.append(f'<p class="title"><b>{link}</b></p>')
    if byline(entry):
        lines.append(f'<p class="byline">{html.escape(byline(entry))}</p>')
    lines.append("</div>")
    return "\n".join(lines)


def edge_section(name: str, target: str, entries: list[dict[str, Any]]) -> str:
    """The edge `name` \u2192 `target`: its header (the confound, an arrow, the variable), the count, the papers."""
    n = len(entries)
    count = (
        f"<b>{papers(n)}</b> {'suggests' if n == 1 else 'suggest'} that {html.escape(name)} affects {html.escape(target)}."
        if n
        else f"No papers found that suggest {html.escape(name)} affects {html.escape(target)}."
    )
    head = (
        f'<div class="edge-head"><span class="chip">{html.escape(name)}</span><span class="edge-arrow">\u2192</span>'
        f'<span class="var-pill">{html.escape(target)}</span></div>'
    )
    body = "\n".join(passage(e, i == 0, name, target) for i, e in enumerate(entries))
    return f'<section class="edge-section">{head}<p class="count">{count}</p>{body}</section>'


def wrap(color: str, inner: str) -> str:
    """Evidence HTML: `--c` is the confound's colour, for the chip, the pill outline and the panel's side bar."""
    return f'<div class="evidence-content" data-color="{color}" style="--c:{color}">{inner}</div>'


def also(name: str, other_target: str, n: int, leg: str) -> str:
    return (
        f'<div class="also">\u2192 Also: <b>{html.escape(name)} \u2192 {html.escape(other_target)}</b> \u00b7 {papers(n)} '
        f"<a href=\"#\" onclick=\"dagLegClick('{js_arg(name)}','{leg}');return false;\">show</a></div>"
    )


def evidence_blob(data: dict[str, Any], confounds: list[dict[str, Any]]) -> dict[str, str]:
    iv, dv = data["iv"], data["dv"]
    blob = {
        "__deselect__": '<div class="evidence-content">Click an arrow in the diagram, or a confound on the left, to see '
        "the papers behind it.</div>"
    }
    for index, confound in enumerate(confounds):
        name, color = confound["name"], COLORS[index % len(COLORS)]
        to_iv, to_dv = yes(confound["evidence_to_iv"]), yes(confound["evidence_to_dv"])
        sections = edge_section(name, iv, to_iv), edge_section(name, dv, to_dv)
        blob[f"__meta__:{name}"] = wrap(color, "".join(sections))
        blob[f"__leg__:{name}:to_iv"] = wrap(color, sections[0] + also(name, dv, len(to_dv), "to_dv"))
        blob[f"__leg__:{name}:to_dv"] = wrap(color, sections[1] + also(name, iv, len(to_iv), "to_iv"))
    return blob


def cards(data: dict[str, Any], confounds: list[dict[str, Any]]) -> str:
    cells = []
    for i, confound in enumerate(confounds):
        name = html.escape(confound["name"])
        buttons = "".join(
            f'<button type="button" class="leg-btn" data-md="{name}" data-leg="{leg}">\u2192 {html.escape(target)} '
            f"<small>{papers(len(yes(confound[key])))}</small></button>"
            for leg, target, key in (
                ("to_iv", data["iv"], "evidence_to_iv"),
                ("to_dv", data["dv"], "evidence_to_dv"),
            )
        )
        cells.append(
            f'<div class="md-card" data-md="{name}" data-color="{COLORS[i % len(COLORS)]}">'
            f'<div class="card-name">{name}</div>{buttons}</div>'
        )
    body = "\n".join(cells) or "<p>No confounders found.</p>"
    return f'<div class="md-card-list"><div class="md-card-list-header">Candidate Confounds</div>{body}</div>'


def js_arg(name: str) -> str:
    """`name` as the inside of a single-quoted JavaScript string in an HTML attribute."""
    return html.escape(name.replace("\\", "\\\\").replace("'", "\\'"), quote=True)


def two_lines(label: str) -> list[str]:
    """A variable's name on at most two lines, split at the word boundary nearest the middle."""
    words = label.split()
    if len(words) < 2:
        return [label]
    cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return [" ".join(words[:cut]), " ".join(words[cut:])]


def unit(frm: tuple[float, float], to: tuple[float, float]) -> tuple[float, float]:
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def node_positions(count: int) -> list[tuple[float, float]]:
    cx, cy = ARC_CENTER
    angles = [0.0] if count == 1 else [-ARC_SPAN + 2 * ARC_SPAN * i / (count - 1) for i in range(count)]
    return [
        (cx + ARC_RADIUS * math.sin(math.radians(a)), cy - ARC_RADIUS * math.cos(math.radians(a)))
        for a in angles
    ]


def edge_svg(name: str, color: str, leg: str, frm: tuple[float, float], variable: str, count: int) -> str:
    """One arrow from a confound's node to a variable. Its paper count is a pill, shown only when the arrow is chosen."""
    target = VARIABLES[variable]
    ux, uy = unit(frm, target)
    x1, y1 = frm[0] + NODE_R * ux, frm[1] + NODE_R * uy
    x2, y2 = target[0] - VARIABLE_R * ux, target[1] - VARIABLE_R * uy
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dash = "" if count else ' stroke-dasharray="6 5"'
    label = html.escape(name)
    where = f'data-md="{label}" data-leg="{leg}"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#888" stroke-width="3" opacity="0.4"{dash} '
        f'marker-end="url(#arrowhead-arc)" style="cursor:pointer" {where} data-color="{color}" '
        f"onclick=\"event.stopPropagation();dagLegClick('{js_arg(name)}','{leg}')\" />\n"
        f'<g class="dag-count pill" {where} transform="translate({mx:.1f},{my:.1f})"><rect x="-38" y="-13" width="76" height="26" rx="13" fill="{color}" />'
        f'<text y="5" text-anchor="middle" font-size="13" font-weight="700" fill="#fff" font-family="IBM Plex Sans, Arial">{papers(count)}</text></g>'
    )


def diagram(data: dict[str, Any], drawn: list[tuple[str, str]], counts: dict[str, tuple[int, int]]) -> str:
    """The SVG: the two variables, and for each drawn confound a node with one arrow to each variable."""
    marker = '<marker id="{id}" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto">'
    marker += '<polygon points="0 0, 6 2.5, 0 5" fill="{fill}" /></marker>'
    defs = marker.format(id="arrowhead", fill="#333") + marker.format(id="arrowhead-arc", fill="#888")
    defs += "".join("\n" + marker.format(id=f"arrow-md-{c[1:]}", fill=c) for _, c in drawn)
    parts = [
        '<svg width="100%" viewBox="0 0 640 480" style="background:white;display:block;margin:0 auto;'
        'max-width:620px;user-select:none;-webkit-user-select:none;" class="dag-svg" onclick="dagBackgroundClick()" >',
        f"<defs>{defs}\n</defs>",
    ]
    nodes = node_positions(len(drawn)) if drawn else []
    for (name, color), at in zip(drawn, nodes):
        n_iv, n_dv = counts[name]
        parts.append(edge_svg(name, color, "to_iv", at, "x", n_iv))
        parts.append(edge_svg(name, color, "to_dv", at, "y", n_dv))
    (x_x, x_y), (y_x, y_y) = VARIABLES["x"], VARIABLES["y"]
    parts.append(
        f'<line x1="{x_x + VARIABLE_R:.1f}" y1="{x_y:.1f}" x2="{y_x - VARIABLE_R:.1f}" y2="{y_y:.1f}" '
        'stroke="#333" stroke-width="2" opacity="1" marker-end="url(#arrowhead)" />'
    )
    parts.append(
        f'<text x="{(x_x + y_x) / 2:.1f}" y="{x_y + 24:.1f}" text-anchor="middle" font-family="IBM Plex Sans, Arial, sans-serif" '
        'font-size="12" font-style="italic" fill="#666">the relationship studied</text>'
    )
    for variable, key, role in (("x", "iv", "cause (IV)"), ("y", "dv", "outcome (DV)")):
        vx, vy = VARIABLES[variable]
        parts.append(
            f'<circle class="dag-ring" data-var="{variable}" cx="{vx:.1f}" cy="{vy:.1f}" r="46" fill="none" stroke="#333" stroke-width="1.5" />'
        )
        parts.append(
            f'<circle class="dag-var" data-var="{variable}" cx="{vx:.1f}" cy="{vy:.1f}" r="{VARIABLE_R}" fill="white" stroke="#333" stroke-width="2.5" />'
        )
        lines = two_lines(data[key])
        for line, text in enumerate(lines):
            parts.append(
                f'<text x="{vx:.1f}" y="{vy + 62 + 15 * line:.1f}" text-anchor="middle" font-family="Arial, Helvetica, '
                f'sans-serif" font-size="13" font-weight="bold" fill="#333">{html.escape(text)}</text>'
            )
        parts.append(
            f'<text x="{vx:.1f}" y="{vy + 62 + 15 * len(lines) + 1:.1f}" text-anchor="middle" font-family="Arial, Helvetica, '
            f'sans-serif" font-size="11.5" fill="#666">{role}</text>'
        )
    for (name, color), (nx, ny) in zip(drawn, nodes):
        label = html.escape(name)
        parts.append(
            f'<circle class="dag-md-circle" cx="{nx:.1f}" cy="{ny:.1f}" r="{NODE_R:.0f}" fill="{color}" fill-opacity="0.3" '
            f'stroke="{color}" stroke-width="2.5" data-md="{label}" data-color="{color}" '
            f"onclick=\"event.stopPropagation();dagMetaClick('{js_arg(name)}')\" />"
        )
        parts.append(
            f'<text class="dag-md-label" data-md="{label}" x="{nx:.1f}" y="{ny - 30:.1f}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="bold" fill="{color}">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def standalone_page(data: dict[str, Any]) -> str:
    """The whole page for a `share_data` dict, as one HTML string with its data, style and script inside."""
    by_name = {c["name"]: c for c in data.get("dag_confounders", [])}
    confounds = [by_name[n] for n in data.get("all_md_names", []) if n in by_name]
    blob = evidence_blob(data, confounds)
    drawn = [(c["name"], COLORS[i % len(COLORS)]) for i, c in enumerate(confounds[:MAX_DRAWN])]
    counts = {c["name"]: (len(yes(c["evidence_to_iv"])), len(yes(c["evidence_to_dv"]))) for c in confounds}
    iv, dv = html.escape(data["iv"]), html.escape(data["dv"])
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ConfoundSearch \u2014 {iv} \u2192 {dv}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
{asset("page.css")}</style>
</head><body>
<header class="app-nav"><div class="brand">Confound<em>Search</em></div></header>
<div class="status-header"><h1>Showing possible confounds for the relationship between <b>{iv}</b> and <b>{dv}</b></h1></div>
<div class="row">
  <div class="col" style="flex:0.62;">{cards(data, confounds)}</div>
  <div class="col" style="flex:1.45;"><div class="dag-container">{diagram(data, drawn, counts)}</div></div>
  <div class="col" style="flex:1.45;"><div class="evidence-box" id="evidence-display">{blob["__deselect__"]}</div></div>
</div>
<script>
var EVIDENCE_BLOB = {json.dumps(blob)};
{asset("page.js")}</script>
</body></html>
"""
