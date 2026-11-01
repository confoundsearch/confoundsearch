"""ConfoundSearch has four layers, one directory each, behind small interfaces. These tests enforce them.

    confoundsearch/data/        where papers come from: a file, Elasticsearch, Semantic Scholar, OpenAlex
    confoundsearch/logic/       the Markov decision process and its audit log
    confoundsearch/model/       the language model: any OpenAI-compatible endpoint
    confoundsearch/interface/   the web page and its backend, the command line, the JSON result, the skill

The rules are about direction. The logic layer knows the *interfaces* of the data and model layers
(`Corpus`, `ChatClient`) and nothing of Elasticsearch, Semantic Scholar, OpenAlex or any particular model.
Data sources and the model layer know nothing of the interface layer or of each other. The page itself
(interface/ui/) knows nothing of the agent. Only the command line, which wires everything together, may see
it all.

A file's layer is the directory it lives in, so moving a file is how you change its layer, and a new file
cannot escape the rules by being unlisted. The few modules at the package root are shared plumbing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import confoundsearch
from confoundsearch.data.corpus import Corpus
from confoundsearch.model.llm import ChatClient

PACKAGE = Path(confoundsearch.__file__).parent
LAYERS = ("logic", "data", "model", "interface")
PLUMBING = {"errors", "http", "dotenv", "runlog"}  # shared, used by the layers that need them
COMPOSITION_ROOT = {"cli", "result", "skill", "__main__"}  # the interface units that wire everything together
BACKEND = {"server", "export"}

# What each layer may import: {layer: {unit: names or None for any}}. A unit is a module or package directly
# inside a layer directory (graph, actions, corpus, ...).
LOGIC_UNITS = {
    "graph",
    "state",
    "actions",
    "transition",
    "reward",
    "policy",
    "episode",
    "audit",
    "expand",
    "deterministic",
}
ALLOWED: dict[str, dict[str, set[str] | None]] = {
    "logic": {"logic": None, "plumbing:errors": None},
    "data": {"data": None, "logic:graph": None, "plumbing:errors": None, "plumbing:http": None},
    "model": {
        "model": None,
        "logic:actions": None,
        "logic:graph": None,
        "logic:state": None,
        "plumbing:errors": None,
        "plumbing:http": None,
    },
}
# The only names the logic layer may take from the data and model layers: their interfaces.
INTERFACES = {"data:corpus": {"Corpus"}, "model:llm": {"ChatClient"}}


def classify(dotted: str) -> tuple[str, str | None] | None:
    """('logic', 'graph') for confoundsearch.logic.graph; ('plumbing', 'errors') for confoundsearch.errors."""
    parts = dotted.split(".")
    if parts[0] != "confoundsearch" or len(parts) == 1:
        return None
    if parts[1] in LAYERS:
        return parts[1], parts[2] if len(parts) > 2 else None
    return "plumbing", parts[1]


def imports(source: str) -> list[tuple[str, str | None, str | None]]:
    """(layer, unit, imported name) for every confoundsearch import, wherever it appears in the file."""
    found: list[tuple[str, str | None, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if target := classify(alias.name):
                    found.append((*target, None))
        elif isinstance(node, ast.ImportFrom) and node.module:
            target = classify(node.module)
            if node.module == "confoundsearch":
                found += [("root", None, alias.name) for alias in node.names]
            elif target:
                found += [(*target, alias.name) for alias in node.names]
    return found


def violations(layer: str, unit: str, source: str) -> list[str]:
    """Why `source`, living in `layer/unit`, breaks the rules. Empty if it does not."""
    bad = []
    for target_layer, target_unit, name in imports(source):
        if target_layer == "root":
            if name != "__version__":
                bad.append(f"imports {name!r} from the package root (only __version__ is allowed)")
            continue
        label = f"{target_layer}.{target_unit}" if target_unit else target_layer
        if not _permitted(layer, unit, target_layer, target_unit, name):
            bad.append(f"imports {label}" + (f".{name}" if name else ""))
    return bad


def _permitted(layer: str, unit: str, target_layer: str, target_unit: str | None, name: str | None) -> bool:
    if layer == "interface":
        if unit == "ui":  # the page: knows only itself
            return target_layer == "interface" and target_unit == "ui"
        if unit in COMPOSITION_ROOT:
            return True
        # the page's backend: the logic layer and the interface layer, never a concrete source or model
        return target_layer in {"logic", "interface"} or (
            target_layer == "plumbing" and target_unit == "errors"
        )
    rules = ALLOWED[layer]
    key = target_layer if target_layer == layer else f"{target_layer}:{target_unit}"
    if key in rules or target_layer == layer:
        return True
    return layer == "logic" and name in INTERFACES.get(key, set())


def units() -> list[tuple[str, str, Path]]:
    found = []
    for layer in LAYERS:
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            relative = path.relative_to(PACKAGE / layer)
            unit = (
                relative.parts[0][: -len(".py")] if relative.parts[0].endswith(".py") else relative.parts[0]
            )
            if unit != "__init__":
                found.append((layer, unit, path))
    return found


def holds_source(directory: Path) -> bool:
    """True if the folder has any real file. One with only compiled bytecode is a leftover, not a layer."""
    return any(
        p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" for p in directory.rglob("*")
    )


def stray_entries(package: Path) -> set[str]:
    """Modules and folders directly inside the package that belong to no layer and are not shared plumbing."""
    stray = {
        p.name
        for p in package.iterdir()
        if p.suffix == ".py" and p.stem not in PLUMBING | {"__init__", "__main__"}
    }
    stray |= {
        p.name
        for p in package.iterdir()
        if p.is_dir() and p.name not in {*LAYERS, "__pycache__"} and holds_source(p)
    }
    return stray


def test_every_python_file_is_in_a_layer_or_is_plumbing():
    """A new top-level module cannot dodge the rules: it must be a layer directory, or shared plumbing."""
    stray = stray_entries(PACKAGE)
    assert not stray, (
        f"put these in one of the layer directories {LAYERS}, or list them as plumbing: {sorted(stray)}"
    )


class TestStrayDetection:
    def package(self, tmp_path):
        for layer in LAYERS:
            (tmp_path / layer).mkdir()
        (tmp_path / "errors.py").write_text("")
        (tmp_path / "__init__.py").write_text("")
        return tmp_path

    def test_a_clean_package_has_no_strays(self, tmp_path):
        assert stray_entries(self.package(tmp_path)) == set()

    def test_a_leftover_folder_holding_only_bytecode_is_not_a_stray(self, tmp_path):
        """What a reorganization leaves behind: git removes the files but not Python's cache."""
        package = self.package(tmp_path)
        (package / "actions" / "__pycache__").mkdir(parents=True)
        (package / "actions" / "__pycache__" / "merge.cpython-312.pyc").write_bytes(b"\0")
        (package / "__pycache__").mkdir()
        assert stray_entries(package) == set()

    def test_an_empty_leftover_folder_is_not_a_stray_either(self, tmp_path):
        package = self.package(tmp_path)
        (package / "old").mkdir()
        assert stray_entries(package) == set()

    @pytest.mark.parametrize("path", ["newlayer/__init__.py", "newlayer/deep/module.py", "newlayer/notes.md"])
    def test_a_folder_with_real_files_is_a_stray(self, tmp_path, path):
        package = self.package(tmp_path)
        (package / path).parent.mkdir(parents=True, exist_ok=True)
        (package / path).write_text("x")
        assert stray_entries(package) == {"newlayer"}

    def test_a_stray_module_is_still_caught_and_plumbing_is_not(self, tmp_path):
        package = self.package(tmp_path)
        (package / "stray.py").write_text("")
        (package / "http.py").write_text("")
        assert stray_entries(package) == {"stray.py"}

    def test_the_real_package_has_none(self):
        assert stray_entries(PACKAGE) == set()


def test_each_layer_directory_is_a_package_that_says_what_it_is():
    for layer in LAYERS:
        init = PACKAGE / layer / "__init__.py"
        assert init.exists() and init.read_text(encoding="utf-8").startswith('"""The ')


@pytest.mark.parametrize(
    "layer,unit,path", units(), ids=lambda v: str(v.relative_to(PACKAGE)) if isinstance(v, Path) else v
)
def test_each_module_imports_only_what_its_layer_allows(layer, unit, path):
    found = violations(layer, unit, path.read_text(encoding="utf-8"))
    assert not found, f"{path.relative_to(PACKAGE)} ({layer} layer) {'; '.join(found)}"


def test_the_logic_layer_has_exactly_the_modules_the_rules_know_about():
    assert {unit for layer, unit, _ in units() if layer == "logic"} == LOGIC_UNITS


def test_the_package_root_only_re_exports():
    """__init__.py gathers the public names, so it is the one file allowed to import from every layer."""
    names = {
        (layer, name) for layer, _, name in imports((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    }
    assert {("logic", "run_episode"), ("data", "ElasticsearchCorpus"), ("model", "LLMJudge")} <= names
    assert ("interface", "save_export") in names


class TestTheInterfacesTheLogicLayerSees:
    @pytest.mark.parametrize("interface", [Corpus, ChatClient], ids=lambda c: c.__name__)
    def test_they_are_protocols_not_implementations(self, interface):
        assert getattr(interface, "_is_protocol", False) is True
        public = [n for n in vars(interface) if not n.startswith("_")]
        assert public == (["search"] if interface is Corpus else ["chat"])  # one method each

    def test_the_judge_and_proposer_interfaces_live_in_the_logic_layer(self):
        from confoundsearch.logic.episode import ActionSource
        from confoundsearch.logic.transition import Judge

        assert getattr(Judge, "_is_protocol", False) and getattr(ActionSource, "_is_protocol", False)

    def test_concrete_implementations_satisfy_them_without_inheriting(self):
        from confoundsearch import (
            ElasticsearchCorpus,
            InMemoryCorpus,
            OpenAICompatibleClient,
            OpenAlexCorpus,
            SemanticScholarCorpus,
        )

        for corpus in (InMemoryCorpus, ElasticsearchCorpus, SemanticScholarCorpus, OpenAlexCorpus):
            assert Corpus not in corpus.__mro__ and callable(corpus.search)
        assert ChatClient not in OpenAICompatibleClient.__mro__ and callable(OpenAICompatibleClient.chat)

    def test_the_public_names_still_come_from_the_package_root(self):
        import confoundsearch as c

        for name in (
            "run_episode",
            "Environment",
            "ElasticsearchCorpus",
            "OpenAICompatibleClient",
            "save_export",
        ):
            assert hasattr(c, name) and name in c.__all__


ES = "from confoundsearch.data.es import ElasticsearchCorpus"
SOURCES = "from confoundsearch.data.sources import SemanticScholarCorpus"
INMEMORY = "from confoundsearch.data.corpus import InMemoryCorpus"
CORPUS = "from confoundsearch.data.corpus import Corpus"
CLIENT = "from confoundsearch.model.llm import OpenAICompatibleClient"
JUDGE = "from confoundsearch.model.llm import LLMJudge"
CHATCLIENT = "from confoundsearch.model.llm import ChatClient"
SERVER = "from confoundsearch.interface.server import make_server"
PAGE = "from confoundsearch.interface.ui import page"
CLI = "from confoundsearch.interface.cli import main"
GRAPH = "from confoundsearch.logic.graph import Passage"
ENV = "from confoundsearch.logic.transition import Environment"
EPISODE = "from confoundsearch.logic.episode import run_episode"
ACTIONS = "from confoundsearch.logic.actions import Retrieve"
STATE = "from confoundsearch.logic.state import State"
AUDIT = "from confoundsearch.logic.audit import AuditLog"
ERRORS = "from confoundsearch.errors import InvalidAction"
HTTP = "from confoundsearch.http import urllib_transport"
DOTENV = "from confoundsearch.dotenv import load_dotenv"

BREACHES = [
    ("logic", "graph", ES),
    ("logic", "graph", SOURCES),
    ("logic", "graph", CLIENT),
    ("logic", "graph", INMEMORY),  # only the interface is allowed
    ("logic", "graph", "import confoundsearch.model.llm"),
    ("logic", "graph", SERVER),
    ("logic", "graph", PAGE),
    ("logic", "graph", CLI),
    ("logic", "graph", "from confoundsearch import OpenAICompatibleClient"),
    ("logic", "graph", "def f():\n    " + HTTP),  # hidden in a function
    ("logic", "graph", "if TYPE_CHECKING:\n    " + JUDGE),  # type-only imports count too
    ("logic", "graph", DOTENV),
    ("data", "corpus", JUDGE),
    ("data", "corpus", ENV),
    ("data", "corpus", SERVER),
    ("data", "corpus", DOTENV),
    ("model", "llm", CORPUS),
    ("model", "llm", ES),
    ("model", "llm", ENV),
    ("model", "llm", PAGE),
    ("interface", "ui", GRAPH),
    ("interface", "ui", EPISODE),
    ("interface", "ui", SERVER),
    ("interface", "server", JUDGE),
    ("interface", "export", ES),
]
LEGITIMATE = [
    ("logic", "graph", CORPUS),  # the interface, by name
    ("logic", "graph", CHATCLIENT),
    ("logic", "graph", GRAPH + "\n" + ERRORS),
    ("logic", "graph", "from confoundsearch import __version__"),
    ("logic", "graph", "import json\nfrom collections.abc import Callable"),  # outside the package
    ("data", "es", GRAPH + "\n" + HTTP),
    ("data", "es", CORPUS),
    ("model", "llm", ACTIONS + "\n" + STATE),
    ("interface", "ui", PAGE),
    ("interface", "server", EPISODE + "\n" + PAGE),
    ("interface", "export", AUDIT + "\n" + PAGE),
    ("interface", "cli", ES + "\n" + JUDGE),
    ("interface", "result", EPISODE),
]


class TestTheCheckerCatchesBreaches:
    """The tests above are only worth anything if the checker would notice a violation."""

    @pytest.mark.parametrize("layer,unit,source", BREACHES)
    def test_a_boundary_breach_is_reported(self, layer, unit, source):
        assert violations(layer, unit, source), f"{layer}/{unit}: {source!r} should have been rejected"

    @pytest.mark.parametrize("layer,unit,source", LEGITIMATE)
    def test_legitimate_imports_pass(self, layer, unit, source):
        assert violations(layer, unit, source) == []

    def test_the_whole_package_root_is_off_limits_apart_from_the_version(self):
        assert violations("logic", "graph", "from confoundsearch import run_episode, __version__") == [
            "imports 'run_episode' from the package root (only __version__ is allowed)"
        ]

    def test_a_plain_import_statement_is_seen_too(self):
        assert imports("import confoundsearch.model.llm as m") == [("model", "llm", None)]
        assert imports("import os, sys") == []
        assert imports("import confoundsearch") == []

    def test_the_message_names_what_was_imported(self):
        assert violations("logic", "graph", "from confoundsearch.data.es import ElasticsearchCorpus") == [
            "imports data.es.ElasticsearchCorpus"
        ]
        assert violations("logic", "graph", "import confoundsearch.model") == ["imports model"]


def test_no_lower_layer_may_use_the_interface_layer_or_each_other():
    """Spelled out: the dependency direction, as the paper describes it."""
    for lower in ("logic", "data", "model"):
        assert not any(key.startswith("interface") for key in ALLOWED[lower]), lower
    assert not any(k.startswith(("data", "model")) for k in ALLOWED["logic"])  # logic: only the interfaces
    assert not any(k.startswith("model") for k in ALLOWED["data"]) and not any(
        k.startswith("data") for k in ALLOWED["model"]
    )
