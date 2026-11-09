"""Where an arrow's paper-count pill goes: pure geometry, tested on its own."""

from __future__ import annotations

import math

import pytest

from confoundsearch.interface.ui.layout import GAP, HEAD_CLEARANCE, PILL_WIDTH, pill_position

MIDDLE = 320.0
RADIUS = 22.0


def test_a_long_arrow_carries_its_pill_at_its_middle():
    assert pill_position((300, 100), (300, 120), (300, 320), MIDDLE, RADIUS) == (300, 220)


def test_an_arrow_just_long_enough_still_carries_it():
    length = PILL_WIDTH + HEAD_CLEARANCE
    assert pill_position((0, 0), (0, 0), (length, 0), MIDDLE, RADIUS) == (length / 2, 0)


@pytest.mark.parametrize(
    "node, expected_x",
    [
        ((135.0, 208.0), 135.0 - (RADIUS + GAP + PILL_WIDTH / 2)),  # left of the midline: further left
        ((505.0, 208.0), 505.0 + (RADIUS + GAP + PILL_WIDTH / 2)),  # right of it: further right
    ],
)
def test_a_short_arrow_puts_its_pill_beside_the_node_on_the_outer_side(node, expected_x):
    start, end = (node[0] + 5, node[1] + 20), (node[0] + 20, node[1] + 60)  # about 43 long
    assert math.hypot(end[0] - start[0], end[1] - start[1]) < PILL_WIDTH + HEAD_CLEARANCE
    assert pill_position(node, start, end, MIDDLE, RADIUS) == (expected_x, node[1])


def test_a_side_pill_clears_the_node():
    x, _ = pill_position((135.0, 208.0), (140, 228), (150, 260), MIDDLE, RADIUS)
    assert 135.0 - x - PILL_WIDTH / 2 >= RADIUS  # the pill's near edge is outside the node's circle
