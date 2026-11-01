"""The paper points readers at the `actions` directory. Keep it, the code map and the code in sync."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import confoundsearch
from confoundsearch.logic import actions
from confoundsearch.logic.actions import ACTIONS

ACTIONS_DIR = Path(actions.__file__).parent
ROOT = Path(__file__).parent.parent
HELPERS = {"__init__", "enumerate"}


def snake(name: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def action_modules() -> list[Path]:
    return sorted(p for p in ACTIONS_DIR.glob("*.py") if p.stem not in HELPERS)


def classes_defined_in(path: Path) -> list[str]:
    return [n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]


def test_one_module_per_action_named_after_it():
    assert sorted(p.stem for p in action_modules()) == sorted(snake(a.__name__) for a in ACTIONS)


@pytest.mark.parametrize("path", action_modules(), ids=lambda p: p.stem)
def test_each_module_defines_exactly_its_action(path):
    (cls,) = classes_defined_in(path)
    assert snake(cls) == path.stem
    assert cls in {a.__name__ for a in ACTIONS}


@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a.__name__)
def test_every_action_carries_its_transition_effect(action):
    assert callable(action.apply)
    assert list(inspect.signature(action.apply).parameters) == ["self", "state", "env"]
    assert action.__doc__ is None or action.__module__.startswith("confoundsearch.logic.actions.")


def test_action_union_matches_registry():
    assert {a.__name__ for a in ACTIONS} == {t.__name__ for t in actions.Action.__args__}


def test_actions_are_exported_from_the_top_level_package():
    for action in ACTIONS:
        assert getattr(confoundsearch, action.__name__) is action


def test_package_docstring_lists_every_action_file():
    doc = actions.__doc__ or ""
    for path in action_modules():
        assert f"{path.stem}.py" in doc
    for action in ACTIONS:
        assert action.__name__ in doc
