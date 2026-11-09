"""Where an arrow's paper-count pill goes in the diagram. Pure geometry: no drawing, no page, no state.

    pill_position(node=(135, 208), start=(141, 229), end=(160, 277), middle_x=320, node_radius=22)

On a long arrow the pill sits on the line, at its middle. A short arrow has no room for the pill clear of its
arrowhead, so there it sits beside the confound's node, on the outer side of the diagram, where no arrow runs.
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]

PILL_WIDTH = 76.0  # as drawn: the pill's rectangle is 76 wide
HEAD_CLEARANCE = 40.0  # room left for the arrowhead and a little air
GAP = 8.0  # between a node's edge and a pill beside it


def pill_position(node: Point, start: Point, end: Point, middle_x: float, node_radius: float) -> Point:
    """The centre of the pill for the arrow from `start` to `end`, which leaves the confound at `node`.

    `middle_x` is the diagram's vertical midline (halfway between the two variables): a node left of it is on
    the left, and its side-pill goes further left; a node right of it, further right.
    """
    if math.hypot(end[0] - start[0], end[1] - start[1]) >= PILL_WIDTH + HEAD_CLEARANCE:
        return (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    outward = -1.0 if node[0] < middle_x else 1.0
    return node[0] + outward * (node_radius + GAP + PILL_WIDTH / 2), node[1]
