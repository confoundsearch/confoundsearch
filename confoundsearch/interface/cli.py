"""Command line: `confoundsearch serve` runs the interface and its backend,
`confoundsearch run` runs one search and writes the standalone page.

    confoundsearch serve --model gpt-4.1                     # the hosted index (ES_URL, ES_API_KEY)
    confoundsearch serve --model gpt-4.1 --corpus openalex   # or semanticscholar, or --corpus-file F
    confoundsearch run --iv "IT investment" --dv "firm performance" --corpus-file passages.jsonl
    confoundsearch audit confoundsearch.logic.audit.jsonl     # check and read a run's audit trail
    confoundsearch run --json ...                        # the result as JSON, for programs and agents
    confoundsearch show known_confounds.json             # turn a JSON blob of known confounds into a page
    confoundsearch setup                                 # write your keys to a .env file
    confoundsearch install-skill                         # add the Claude Code skill
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from confoundsearch import __version__
from confoundsearch.data.cache import (
    CachedCorpus,
    PaperDatabase,
    cache_dir,
    databases,
    flush,
    wait_for_writes,
)
from confoundsearch.data.corpus import Corpus, InMemoryCorpus
from confoundsearch.data.demo import (
    CURATED_CONFOUNDS,
    CURATED_SEARCH,
    DEMO_QUESTION,
    PREWARMED_NOTE,
    RECORDED_MODEL,
    DemoCorpus,
    seed_demo_cache,
)
from confoundsearch.data.es import ElasticsearchCorpus
from confoundsearch.data.keyword_service import HttpCorpus, KeywordService, service_url
from confoundsearch.data.multi import MultiCorpus
from confoundsearch.data.sources import OpenAlexCorpus, SemanticScholarCorpus
from confoundsearch.dotenv import keep_out_of_git, load_dotenv, update_dotenv
from confoundsearch.errors import BudgetReached, RemoteError, ReplayMiss
from confoundsearch.http import Transport, resilient, urllib_transport
from confoundsearch.interface.expander import build_expander
from confoundsearch.interface.export import to_share_data
from confoundsearch.interface.known import load_known, page_from_json
from confoundsearch.interface.modelcache import CachedClient, find_cached
from confoundsearch.interface.result import run_result
from confoundsearch.interface.review import FORMATS as REVIEW_FORMATS
from confoundsearch.interface.review import load_share, open_page, review_text, where_it_went
from confoundsearch.interface.server import RunManager, make_server
from confoundsearch.interface.ui import standalone_html
from confoundsearch.logic.actions import AddConfound, AddEdge, AttachPassage, Merge, Retrieve
from confoundsearch.logic.audit import AuditedClient, AuditLog, AuditTrail
from confoundsearch.logic.audit import current as current_log
from confoundsearch.logic.deterministic import DeterministicPolicy
from confoundsearch.logic.episode import ActionSource, Step, Trajectory, run_episode
from confoundsearch.logic.expand import ExpandedCorpus
from confoundsearch.logic.policy import EpsilonGreedy, LookaheadQ, Policy
from confoundsearch.logic.reward import phi
from confoundsearch.logic.transition import Environment, Judge, TrustSynthesis
from confoundsearch.model.embeddings import OpenAIEmbedder
from confoundsearch.model.expansion import FIELDS, LLMMeasures, LLMMechanisms
from confoundsearch.model.llm import (
    OPENROUTER_URL,
    ChatClient,
    LLMConfounds,
    LLMJudge,
    LLMProposer,
    LLMRewordings,
    LLMSynonyms,
    LLMSynthesis,
    OpenAICompatibleClient,
    Proposer,
    temperature_for,
)
from confoundsearch.runlog import RunLog

EPIGRAPH = (
    "\u201c[T]he theorist is overloaded by demands . . . "
    "in a head that suffers from bounded rationality.\u201d"
    "\n  \u2014 Karl Weick, Theory Construction as Disciplined Imagination (1989, p. 529)"
)


def describe(action: object) -> str:
    """An action in the paper's words: the move, and the causal claim it makes about X and Y."""
    if isinstance(action, Retrieve):
        return f"Retrieve({action.query!r})"
    if isinstance(action, AddConfound):
        return f"AddConfound({action.label!r})  a candidate C"
    if isinstance(action, AddEdge):
        return f"AddEdge({action.confound!r})  C \u2192 {action.target}"
    if isinstance(action, AttachPassage):
        return (
            f"AttachPassage({action.confound!r}, {action.passage_id})  evidence for C \u2192 {action.target}"
        )
    if isinstance(action, Merge):
        return f"Merge({action.a!r}, {action.b!r})  one construct, two names"
    return "Stop()"


def progress(number: int, step: Step) -> str:
    """One step as a status update in Claude Code's style: the action a_t, then r_t and the score \u03a6."""
    graph = step.next_state.graph
    return (
        f"\u25cf a{number} = {describe(step.action)}\n"
        f"  \u23bf r{number} = {step.reward:+.2f}, \u03a6 = {phi(graph):.2f}, "
        f"{len(graph.connected())} confounds with both X \u2190 C \u2192 Y"
    )


def status_text(event: dict[str, Any]) -> str | None:
    """What a logged event means to someone watching: what is being searched, or why it is waiting."""
    kind, url = event["event"], str(event.get("url", ""))
    if kind in ("request", "http"):
        parts = urlsplit(url)
        asked = parse_qs(parts.query)
        what = next((v[0] for k, v in asked.items() if k in ("query", "search", "q")), "")
        where = parts.netloc.replace("api.", "", 1) or "the search service"
        if kind == "request":
            retry = f" (try {event['attempt']})" if event["attempt"] > 1 else ""
            return f"searching {where}{f' for {what!r}' if what else ''}{retry}"
        if event.get("retrying_in") is not None:
            return (
                f"{where} said HTTP {event['status']}; waiting {event['retrying_in']:g}s, then trying again"
            )
        if event["status"] != 200:
            return f"{where} said HTTP {event['status']}; giving up on this search"
    if kind == "source_failed":
        return f"{event['source']} failed, carrying on with the other sources"
    if kind == "source_skipped":
        return f"{event['source']} is resting after a failure; trying it again in {event['retry_in']:g}s"
    if kind == "source_recovered":
        return f"{event['source']} is answering again"
    return None


class StatusLine:
    """What the run is doing right now: one line on stderr, rewritten in place (like Claude Code's), with
    a spinner and the time elapsed. It shows the current activity, step and score, and prints nothing
    else. Where stderr is not a terminal (a pipe, a file) it cannot be rewritten, so it stays quiet except
    for rate limits and failed sources, one line each."""

    FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(
        self, stream: Any = None, clock: Callable[[], float] = time.monotonic, interval: float = 0.2
    ) -> None:
        self.stream = stream or sys.stderr
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.clock, self.interval, self.began = clock, interval, clock()
        self.activity, self.step, self.total, self.phi = "starting", 0, 0, 0.0
        self.shown, self.frame = False, 0
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        kind, text = event["event"], status_text(event)
        with self.lock:
            if kind == "step":
                self.step, self.phi = event["n"], event.get("phi", self.phi)
                self.activity = "choosing the next move"
            elif kind == "asking_model":
                self.activity = "asking the model"
            elif text:
                self.activity = text
            if self.live:
                self._draw()
            elif text and (kind == "source_failed" or event.get("status", 200) != 200):
                self.stream.write(f"  {text}\n")
                self.stream.flush()

    def _draw(self) -> None:
        seconds = int(self.clock() - self.began)
        spin = self.FRAMES[self.frame % len(self.FRAMES)]
        self.frame += 1
        steps = f"step {self.step}/{self.total}" if self.total else f"step {self.step}"
        clock = f"{seconds // 60}:{seconds % 60:02d}"
        line = f"{spin} {self.activity} \u2026 ({clock} \u00b7 {steps} \u00b7 \u03a6 {self.phi:.2f})"
        width = shutil.get_terminal_size((100, 20)).columns - 1
        self.stream.write("\r\x1b[K" + (line if len(line) <= width else line[: width - 1] + "\u2026"))
        self.stream.flush()
        self.shown = True

    def start(self) -> None:
        """Keep the clock and spinner moving on a terminal, even while one long request is pending."""
        if self.live and self.thread is None:
            self.thread = threading.Thread(target=self._tick, daemon=True)
            self.thread.start()

    def _tick(self) -> None:
        while not self.stopped.wait(self.interval):
            with self.lock:
                self._draw()

    def stop(self) -> None:
        """Stop updating and wipe the line, so what is printed next starts on a clean line."""
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        with self.lock:
            if self.shown:
                self.stream.write("\r\x1b[K")
                self.stream.flush()
                self.shown = False


class Announced:
    """A chat client that tells the run log before each call, so the status line can say it is waiting, and
    refuses the next call once the estimated spend has reached `budget` dollars."""

    def __init__(
        self, client: ChatClient, log: Callable[..., None] | None, budget: float | None = None
    ) -> None:
        self.client, self.log, self.budget = client, log, budget
        self.model = getattr(client, "model", None)
        self.spend = getattr(client, "spend", None)  # the client's own count of what it has cost

    @property
    def dollars(self) -> float:
        return float(self.spend.dollars) if self.spend is not None else 0.0

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        if self.budget is not None and self.dollars >= self.budget:
            raise BudgetReached(
                f"the language model has cost about ${self.dollars:.2f}, the budget of ${self.budget:.2f}"
            )
        if self.log:
            self.log("asking_model", model=self.model)
        return self.client.chat(messages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="confoundsearch", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    llm = shared.add_argument_group("language model (any OpenAI-compatible endpoint)")
    llm.add_argument("--model", required=True)
    llm.add_argument("--base-url", help="default: $OPENAI_BASE_URL or https://api.openai.com/v1")
    llm.add_argument("--api-key", help="default: $OPENAI_API_KEY (with --openrouter: $OPENROUTER_API_KEY)")
    llm.add_argument(
        "--openrouter", action="store_true", help="use OpenRouter; --model is a slug like openai/gpt-4.1"
    )
    llm.add_argument(
        "--policy",
        choices=POLICIES,
        default="deterministic",
        help="deterministic (default): ask for confounds, get synonyms, search for all, then process; "
        "or epsilon-greedy, a learned-style explorer that picks one move at a time",
    )
    llm.add_argument(
        "--confounds",
        type=int,
        default=8,
        metavar="N",
        help="deterministic policy: how many confounds to ask for",
    )
    llm.add_argument(
        "--no-synthesis",
        action="store_true",
        help="deterministic: skip the synthesis (one reading of everything retrieved, by meaning) and keep "
        "only passages that use the confound's and the variable's words",
    )
    llm.add_argument(
        "--no-mechanisms",
        action="store_true",
        help="deterministic: skip the second round of searches on the measures of each variable and the "
        "mechanisms of each confound (it needs the synthesis)",
    )
    llm.add_argument(
        "--queries",
        type=int,
        default=3,
        metavar="N",
        help="deterministic: searches per confound and variable, plainest wording first (default 3; "
        "at most (synonyms + 1) squared pairs exist)",
    )
    llm.add_argument(
        "--synonyms",
        type=int,
        metavar="N",
        help="deterministic: synonyms per term (default 3); epsilon-greedy: also search N rewordings of each "
        "query and merge the results (default 0)",
    )
    env = os.environ.get
    papers = shared.add_argument_group("papers to search (default: every source set up, merged)")
    papers.add_argument(
        "--corpus",
        action="append",
        metavar="SOURCE",
        help="where to search: service, elasticsearch, semanticscholar, openalex, file, demo (a few hundred "
        "shipped papers, searched offline), or all (the default: the keyword service if --api is set, the "
        "hosted index if ES_URL is set, plus Semantic Scholar and OpenAlex). Repeat it, or separate with "
        "commas, to search several at once; results are merged and duplicates removed, and a source that "
        "is down is skipped",
    )
    papers.add_argument("--corpus-file", type=Path, help="JSONL with id, text, and optionally title, url")
    papers.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="the data directory, where searches are saved so they are made only once (default: "
        "$CONFOUNDSEARCH_CACHE, else a folder in the system's temporary directory); empty it with "
        "`confoundsearch flush`",
    )
    papers.add_argument("--no-cache", action="store_true", help="always search; save nothing")
    papers.add_argument(
        "--api",
        help="address of the ConfoundSearch service, which searches papers for you with no key of yours "
        "(default: $CONFOUNDSEARCH_API, else the address built into confoundsearch, if it has one). Your "
        "search terms are sent to it. `--api off` switches it off",
    )
    papers.add_argument("--es-url", default=env("ES_URL"), help="Elasticsearch endpoint (default: $ES_URL)")
    papers.add_argument("--es-index", default="papers")
    papers.add_argument("--es-api-key", default=env("ES_API_KEY"), help="default: $ES_API_KEY")
    papers.add_argument(
        "--s2-key", default=env("S2_KEY"), help="Semantic Scholar API key (optional; $S2_KEY)"
    )
    papers.add_argument(
        "--openalex-key", default=env("OPENALEX_API_KEY"), help="OpenAlex API key ($OPENALEX_API_KEY)"
    )
    papers.add_argument(
        "--openalex-email", default=env("OPENALEX_EMAIL"), help="email for OpenAlex ($OPENALEX_EMAIL)"
    )

    serve = commands.add_parser("serve", parents=[shared], help="serve the interface and its backend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--max-concurrent", type=int, default=2)
    serve.add_argument("--audit-dir", type=Path, help="also write every run's audit trail to this folder")

    run = commands.add_parser("run", parents=[shared], help="run one search and write the standalone page")
    run.add_argument("--iv", required=True)
    run.add_argument("--dv", required=True)
    run.add_argument(
        "--max-steps", type=int, help="most moves to make (default: deterministic 1000, epsilon-greedy 40)"
    )
    run.add_argument(
        "--max-budget",
        type=float,
        metavar="USD",
        help="stop once the language model has cost about this many dollars (an estimate from tokens used); "
        "what was found so far is kept",
    )
    run.add_argument(
        "--replay",
        action="store_true",
        help="answer every model call from saved replies, asking no model (so no key is needed): only for a "
        "question whose replies are saved, such as the built-in example",
    )
    run.add_argument("--epsilon", type=float, default=0.1, help="epsilon-greedy only")
    run.add_argument("--seed", type=int, help="epsilon-greedy only: makes its choices repeatable")
    run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write the result page (the audit trail goes next to it)",
    )
    run.add_argument("--audit", type=Path, help="where to write the audit trail (default: next to --out)")
    run.add_argument(
        "--log",
        type=Path,
        help="where to write the run log: every search request and step (default: next to --out)",
    )
    run.add_argument(
        "--json", action="store_true", help="print the result as JSON (and nothing else) to stdout"
    )
    run.add_argument(
        "--review",
        nargs="?",
        const="ui",
        choices=REVIEW_FORMATS,
        metavar="FORMAT",
        help="when done, review the page just written: ui (the default) opens it, text prints it here",
    )

    demo = commands.add_parser(
        "demo",
        help="try it: a short example run on the built-in question (organizational structure and firm "
        "performance), then open the result. Its searches ship pre-saved, so it is fast",
    )
    demo.add_argument(
        "--model", help=f"default: {RECORDED_MODEL}, or openai/{RECORDED_MODEL} with only an OpenRouter key"
    )
    demo.add_argument("--base-url", help="default: $OPENAI_BASE_URL or https://api.openai.com/v1")
    demo.add_argument("--api-key", help="default: $OPENAI_API_KEY (with --openrouter: $OPENROUTER_API_KEY)")
    demo.add_argument("--openrouter", action="store_true", help="use OpenRouter")
    demo.add_argument("--out", type=Path, default=Path("confoundsearch-demo.html"))
    demo.add_argument(
        "--confounds",
        type=int,
        metavar="N",
        help="how many confounds to look for (default: 3, the curated example's; 4 with --live)",
    )
    demo.add_argument("--max-budget", type=float, default=1.0, metavar="USD", help="stop at about this cost")
    demo.add_argument("--no-open", action="store_true", help="write the page but do not open it")
    demo.add_argument(
        "--live",
        action="store_true",
        help="search the real sources instead of the shipped example papers",
    )

    expand = commands.add_parser(
        "expand",
        help="turn a construct into a long list of query terms (model synonyms and nearest keywords)",
    )
    expand.add_argument("construct", help='the idea to expand, e.g. "firm size"')
    expand.add_argument("--model", help="default: a small cheap model that fits the key that is set")
    expand.add_argument("--base-url", help="default: $OPENAI_BASE_URL or https://api.openai.com/v1")
    expand.add_argument("--api-key", help="default: $OPENAI_API_KEY (with --openrouter: $OPENROUTER_API_KEY)")
    expand.add_argument("--openrouter", action="store_true", help="use OpenRouter")
    expand.add_argument(
        "--api",
        help="address of the ConfoundSearch service that holds the keyword vectors "
        "(default: $CONFOUNDSEARCH_API, else the address built into confoundsearch)",
    )
    expand.add_argument(
        "--rounds", type=int, default=5, help="times to ask the model, each from another angle"
    )
    expand.add_argument("--per-round", type=int, default=12, help="terms asked for in each round")
    expand.add_argument(
        "--fields",
        default=",".join(FIELDS),
        help="fields to ask the model about, comma separated, or `none` (default: nine social sciences)",
    )
    expand.add_argument("--per-field", type=int, default=8, help="terms asked for from each field")
    expand.add_argument(
        "--per-variant", type=int, default=4, help="synonyms asked for of each field term (0: none)"
    )
    expand.add_argument("--per-hypernym", type=int, default=3, help="broader terms asked for (0: none)")
    expand.add_argument(
        "--per-mechanism", type=int, default=6, help="channels it works through asked for (0: none)"
    )
    expand.add_argument(
        "--neighbor-top", type=int, default=100, help="how many of the best keywords to ask synonyms of"
    )
    expand.add_argument(
        "--per-neighbor", type=int, default=2, help="synonyms asked for of each of those (0: none)"
    )
    expand.add_argument("--max-seeds", type=int, default=150, help="most terms to fetch neighbours for")
    expand.add_argument("--per-term", type=int, default=100, help="nearest keywords fetched for each term")
    expand.add_argument("--limit", type=int, default=1000, help="most terms to return")
    expand.add_argument(
        "--min-works", type=int, default=0, help="only keywords used by at least this many papers"
    )
    expand.add_argument(
        "--context", default="", help="a sentence of setting for the model, e.g. what is studied"
    )
    expand.add_argument("--json", action="store_true", help="print the terms as JSON")
    expand.add_argument("--out", type=Path, help="also write the full result here as JSON")

    audit = commands.add_parser("audit", help="check and read an audit trail written by a run")
    audit.add_argument("file", type=Path, help="an audit trail (.jsonl)")
    audit.add_argument(
        "--json", action="store_true", help="print a machine-readable summary instead of the report"
    )

    show = commands.add_parser("show", help="turn a JSON file of known confounds into the standalone page")
    show.add_argument("file", type=Path, help="known confounds (JSON), or a share_data.json from a run")
    show.add_argument("--out", type=Path, help="default: the file's name with .html")
    review = commands.add_parser("review", help="look at a result: open the interface, or read it as text")
    review.add_argument(
        "--in",
        dest="input",
        type=Path,
        required=True,
        help="what `run --out` wrote (.html), or a .zip or .json export",
    )
    review.add_argument(
        "--format", choices=REVIEW_FORMATS, default="ui", help="ui opens a browser (default); text prints"
    )
    flush_cmd = commands.add_parser("flush", help="delete the searches saved on this machine")
    flush_cmd.add_argument(
        "--cache-dir",
        type=Path,
        help="default: $CONFOUNDSEARCH_CACHE, else a folder in the system's temporary directory",
    )
    flush_cmd.add_argument("--iv", help="only this question's database (with --dv); default: all of them")
    flush_cmd.add_argument("--dv")
    setup = commands.add_parser("setup", help="ask for your API keys and write them to a .env file")
    setup.add_argument("--env-file", type=Path, default=Path(".env"))
    skill = commands.add_parser(
        "install-skill", help="install the Claude Code skills: /confoundsearch and /confoundsearchui"
    )
    skill.add_argument("--dir", type=Path, help="skills folder (default: ~/.claude/skills)")
    skill.add_argument("--project", action="store_true", help="install into ./.claude/skills instead")
    skill.add_argument("--force", action="store_true", help="overwrite a different SKILL.md")
    skill.add_argument(
        "--print",
        dest="show",
        action="store_true",
        help="print the /confoundsearch skill instead of installing",
    )
    return parser


CORPORA = ("service", "elasticsearch", "semanticscholar", "openalex", "file", "demo")
S2_INTERVAL = 1.1  # seconds between Semantic Scholar requests: its keyed limit is one a second
OPENALEX_INTERVAL = 0.2
POLICIES = ("deterministic", "epsilon-greedy")
SYNTHESIS_ATTACH = 8  # passages cited per link when the synthesis has already chosen them (4 when it has not)
RUN_STEPS = 1000  # a deterministic run ends by itself; this only stops one that would never end


def corpus_names(args: argparse.Namespace) -> list[str]:
    """The sources asked for: repeated or comma-separated --corpus, `all` meaning every one that is set up."""
    asked = [name.strip() for chunk in args.corpus or [] for name in chunk.split(",") if name.strip()]
    names = asked or (["file"] if args.corpus_file else ["all"])
    unknown = [n for n in names if n not in (*CORPORA, "all")]
    if unknown:
        raise SystemExit(f"error: unknown source {unknown[0]!r}; choose from {', '.join((*CORPORA, 'all'))}")
    if args.corpus_file and "file" not in names:
        raise SystemExit("error: --corpus-file goes with --corpus file, not with another source")
    if "file" in names and not args.corpus_file:
        raise SystemExit("error: --corpus file needs --corpus-file PATH")
    everything = (
        ["service"] * bool(service_url(args.api))
        + ["elasticsearch"] * bool(args.es_url)
        + ["semanticscholar", "openalex"]
    )
    expanded = [n for name in names for n in (everything if name == "all" else [name])]
    wanted = list(dict.fromkeys(expanded))
    if "service" in wanted and not service_url(args.api):
        raise SystemExit(
            "error: the service has no address (or is switched off): give --api URL, or CONFOUNDSEARCH_API "
            "in .env; or pick another source: --corpus semanticscholar, --corpus openalex, --corpus-file PATH"
        )
    if "elasticsearch" in wanted and not args.es_url:
        raise SystemExit(
            "error: the hosted Elasticsearch index needs ES_URL and ES_API_KEY (put them in a .env file), "
            "or pick another source: --corpus semanticscholar, --corpus openalex, or --corpus-file PATH"
        )
    return wanted


def build_corpus(args: argparse.Namespace) -> Corpus:
    """The papers to search: one source, or several searched at once, merged and de-duplicated."""
    runlog = getattr(args, "runlog", None)
    log = runlog.event if runlog else None

    def paced(interval: float) -> Transport:  # a courteous pace, and another try when a source says wait
        return resilient(urllib_transport, log, min_interval=interval)

    makers: dict[str, Callable[[], Corpus]] = {
        "file": lambda: InMemoryCorpus.from_jsonl(args.corpus_file),
        "demo": DemoCorpus,
        "service": lambda: HttpCorpus(service_url(args.api)),
        "semanticscholar": lambda: SemanticScholarCorpus(args.s2_key, transport=paced(S2_INTERVAL)),
        "openalex": lambda: OpenAlexCorpus(
            args.openalex_key, args.openalex_email, transport=paced(OPENALEX_INTERVAL)
        ),
        "elasticsearch": lambda: ElasticsearchCorpus(
            args.es_url, args.es_index, args.es_api_key, transport=paced(0.0)
        ),
    }
    folder = args.cache_dir or cache_dir()
    stores = {
        name: (
            store
            if name in ("file", "demo") or args.no_cache
            else CachedCorpus(store, folder, name, log, current_question)
        )
        for name, store in ((name, makers[name]()) for name in corpus_names(args))
    }
    if len(stores) == 1:
        return next(iter(stores.values()))
    return MultiCorpus(stores, log)


def build_client(args: argparse.Namespace) -> OpenAICompatibleClient:
    """The AI model: OpenRouter with --openrouter, or when OPENROUTER_API_KEY is the only key you have set.

    Otherwise any OpenAI-compatible server: --base-url or OPENAI_BASE_URL, else OpenAI, with OPENAI_API_KEY.
    """
    only_openrouter = os.environ.get("OPENROUTER_API_KEY") and not (
        args.api_key or args.base_url or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")
    )
    if not (args.openrouter or only_openrouter):
        return OpenAICompatibleClient(
            args.model, base_url=args.base_url, api_key=args.api_key, temperature=temperature_for(args.model)
        )
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("error: --openrouter needs OPENROUTER_API_KEY (put it in a .env file) or --api-key")
    return OpenAICompatibleClient(
        args.model,
        base_url=args.base_url or OPENROUTER_URL,
        api_key=key,
        temperature=temperature_for(args.model),
    )


@dataclass
class Backend:
    """What a run needs: the environment, where open-ended moves come from, and the (audited) model."""

    env: Environment
    proposer: ActionSource
    client: ChatClient


def synonym_count(args: argparse.Namespace) -> int:
    if args.synonyms is not None:
        return int(args.synonyms)
    return 3 if args.policy == "deterministic" else 0


def describe_run(args: argparse.Namespace) -> dict[str, Any]:
    """What a run was asked to do, for the log. Keys are reported as present or not, never shown."""
    key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    return {
        "package": __version__,
        "python": ".".join(map(str, sys.version_info[:3])),
        "platform": sys.platform,
        "iv": args.iv,
        "dv": args.dv,
        "model": args.model,
        "model_host": urlsplit(args.base_url or "").netloc or None,
        "policy": args.policy,
        "confounds": args.confounds,
        "synonyms": synonym_count(args),
        "queries": args.queries,
        "max_steps": args.max_steps,
        "max_budget": args.max_budget,
        "cache": None if args.no_cache else str(args.cache_dir or cache_dir()),
        "replay": bool(getattr(args, "replay", False)),
        "sources": corpus_names(args) if not args.corpus_file or args.corpus else ["file"],
        "keys_present": {
            "model": bool(key),
            "semantic_scholar": bool(args.s2_key),
            "elasticsearch": bool(args.es_url and args.es_api_key),
            "openalex": bool(args.openalex_key or args.openalex_email),
        },
    }


def build_backend(args: argparse.Namespace) -> Backend:
    runlog = getattr(args, "runlog", None)
    log = runlog.event if runlog else None
    asked = None if getattr(args, "replay", False) else build_client(args)  # replaying needs no model, no key
    # the model's answers are saved too (unless --no-cache), so the same question is never paid for twice
    chat: ChatClient = (
        asked
        if args.no_cache and asked is not None
        else CachedClient(asked, args.cache_dir or cache_dir(), current_question, log, model=args.model)
    )
    client = AuditedClient(Announced(chat, log, getattr(args, "max_budget", None)))
    corpus = build_corpus(args)
    if (
        args.policy == "deterministic"
    ):  # it carries its own plan: no model call to propose moves at every step
        judge: Judge = LLMJudge(client) if args.no_synthesis else TrustSynthesis()  # see TrustSynthesis
        return Backend(Environment(corpus, judge), Proposer(), client)
    if synonym_count(args) > 0:
        corpus = ExpandedCorpus(corpus, LLMRewordings(client, synonym_count(args)), synonym_count(args))
    return Backend(Environment(corpus, LLMJudge(client)), Proposer(LLMProposer(client)), client)


def deterministic_policy(args: argparse.Namespace, client: ChatClient) -> DeterministicPolicy:
    return DeterministicPolicy(
        LLMConfounds(client, args.confounds),
        LLMSynonyms(client, synonym_count(args)),
        max_confounds=args.confounds,
        synonyms_per_term=synonym_count(args),
        queries_per_side=args.queries,
        attach_per_side=4 if args.no_synthesis else SYNTHESIS_ATTACH,
        synthesize=None if args.no_synthesis else LLMSynthesis(client),
        measures=None if args.no_synthesis or args.no_mechanisms else LLMMeasures(client),
        mechanisms=None if args.no_synthesis or args.no_mechanisms else LLMMechanisms(client),
    )


def build_policy(args: argparse.Namespace, backend: Backend) -> Policy:
    if args.policy == "deterministic":
        return deterministic_policy(args, backend.client)
    return EpsilonGreedy(
        LookaheadQ(backend.env.preview), args.epsilon, random.Random(args.seed), explore_stop=False
    )


def review(path: Path, format: str) -> int:
    """`confoundsearch review --in FILE`: open the interface on a result, or read it as text."""
    try:
        data = load_share(path)
        if data is None and format == "text":
            raise ValueError("reading it as text needs the .zip or .json, not the page")
        if format == "text":
            print(review_text(data or {}), end="")
            return 0
        page = path
        if data is not None:
            page = path.with_suffix(".html")
            page.write_text(page_from_json(data), encoding="utf-8")
    except (ValueError, OSError, KeyError) as err:
        print(f"error: {path}: {err}", file=sys.stderr)
        return 1
    open_page(page)
    return 0


SERVICES = {"1": "OPENROUTER_API_KEY", "2": "OPENAI_API_KEY"}


def current_question() -> tuple[str, str] | None:
    """The independent and dependent variable of the run this thread is part of, if there is one."""
    log = current_log()
    for event in log.events if log is not None else ():
        if event.kind == "run_started":
            return str(event.data["independent_variable"]), str(event.data["dependent_variable"])
    return None


DEMO_IV, DEMO_DV = DEMO_QUESTION


def demo_model(args: argparse.Namespace) -> str | None:
    """The model for the demo: the one asked for, else a small cheap one that fits the key that is set."""
    if args.model:
        return str(args.model)
    only_openrouter = os.environ.get("OPENROUTER_API_KEY") and not (
        args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")
    )
    if args.openrouter or only_openrouter:
        return f"openai/{RECORDED_MODEL}"
    if args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return RECORDED_MODEL
    return None


def demo(args: argparse.Namespace) -> int:
    """Run the example question through `run`: the curated example, or the real sources with --live."""
    model = demo_model(args)
    replay = model is None  # no key: replay the recorded run, which needs no model
    if replay:
        print(
            "No language model key found, so this replays the recorded example run (the model's answers\n"
            "ship with the package). To run it live, add a key: `confoundsearch setup`.\n",
            file=sys.stderr,
        )
        model = RECORDED_MODEL
    argv = ["run", "--model", str(model), "--iv", DEMO_IV, "--dv", DEMO_DV]
    if not args.live:
        argv += ["--corpus", "demo", *CURATED_SEARCH]  # the shipped example papers: no internet needed
    elif replay:
        argv += ["--corpus", "semanticscholar,openalex"]  # the sources whose searches ship saved
    argv += ["--replay"] if replay else []
    confounds = args.confounds or (4 if args.live else len(CURATED_CONFOUNDS))
    argv += ["--confounds", str(confounds), "--max-budget", str(args.max_budget), "--out", str(args.out)]
    for flag, value in (("--base-url", args.base_url), ("--api-key", args.api_key)):
        argv += [flag, value] if value else []
    argv += ["--openrouter"] if args.openrouter else []
    argv += [] if args.no_open else ["--review"]
    return main(argv)


def flush_saved(args: argparse.Namespace) -> int:
    """Empty the saved searches: all of them, or those of one --iv/--dv question."""
    if (args.iv is None) != (args.dv is None):
        print(
            "error: give both --iv and --dv to empty one question, or neither to empty everything",
            file=sys.stderr,
        )
        return 1
    folder = args.cache_dir or cache_dir()
    question = (args.iv, args.dv) if args.iv is not None else None
    replies = sum(
        PaperDatabase(path).call_count() for path in databases(folder, question)
    )  # counted before they go
    searches, papers, freed = flush(folder, question)
    print(
        f"removed {searches} saved searches, {papers} papers and {replies} model replies "
        f"({freed / 1_000_000:.1f} MB) from {folder}"
    )
    return 0


def setup(args: argparse.Namespace) -> int:
    """Ask for the language-model key and, optionally, a Semantic Scholar key; keep them in a private .env."""
    choice = input("Language model: 1) OpenRouter  2) OpenAI [1]: ").strip() or "1"
    if choice not in SERVICES:
        print(f"error: choose 1 or 2, not {choice!r}", file=sys.stderr)
        return 1
    values = {}
    for name, required in ((SERVICES[choice], True), ("S2_KEY", False)):
        label = name if required else f"{name} (Semantic Scholar, optional; Enter to skip)"
        value = getpass.getpass(f"{label}: ").strip()  # not echoed
        if value:
            values[name] = value
    if not values:
        print("error: no key given; nothing written", file=sys.stderr)
        return 1
    here = cache_dir()
    print(
        "\nConfoundSearch downloads papers as it searches and keeps them in a data directory, like Hugging\n"
        f"Face's model folder. Now: {here}\n"
        "Unless you choose another folder this is in your system's temporary directory, which is emptied\n"
        "from time to time. Empty it yourself any time with `confoundsearch flush`."
    )
    folder = input("Folder to keep downloaded papers in (Enter to leave it as it is): ").strip()
    if folder:
        values["CONFOUNDSEARCH_CACHE"] = str(Path(folder).expanduser().resolve())
    update_dotenv(args.env_file, values)
    print(f"wrote {', '.join(values)} to {args.env_file} (not shown here)")
    if keep_out_of_git(args.env_file):
        print(f"added {args.env_file.name} to .gitignore, so it is not committed")
    return 0


SKILLS = {"confoundsearch": "SKILL.md", "confoundsearchui": "UI_SKILL.md"}  # skill name -> packaged file


def skill_text(name: str = "confoundsearch") -> str:
    return (Path(__file__).resolve().parent / "skill" / SKILLS[name]).read_text(encoding="utf-8")


def install_skill(args: argparse.Namespace) -> int:
    """Copy the packaged skills to <skills folder>/<name>/SKILL.md, where Claude Code finds them."""
    if args.show:
        print(skill_text(), end="")
        return 0
    default = Path.cwd() / ".claude" / "skills" if args.project else Path.home() / ".claude" / "skills"
    status = 0
    for name in SKILLS:
        text, target = skill_text(name), (args.dir or default) / name / "SKILL.md"
        if target.exists() and target.read_text(encoding="utf-8") == text:
            print(f"already installed: {target}")
        elif target.exists() and not args.force:
            print(f"error: {target} exists and differs; use --force to overwrite it", file=sys.stderr)
            status = 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"installed {target}")
    return status


def expand(args: argparse.Namespace) -> int:
    """`confoundsearch expand "firm size"`: a long list of terms, recall first and precision later."""
    model = demo_model(args)
    if model is None:
        print(
            "error: give --model, or set OPENAI_API_KEY (or OPENROUTER_API_KEY) in a .env file",
            file=sys.stderr,
        )
        return 1
    address = service_url(args.api)
    if not address:
        print(
            "error: the keyword service has no address (or is switched off): give --api URL", file=sys.stderr
        )
        return 1
    args.model = model
    service = KeywordService(
        address, embedder=OpenAIEmbedder()
    )  # a non-keyword phrase is embedded with your key
    expander = build_expander(
        build_client(args),
        service,
        args.rounds,
        args.per_round,
        args.context,
        args.min_works,
        args.per_term,
        args.limit,
        fields=()
        if args.fields.strip().casefold() == "none"
        else tuple(f.strip() for f in args.fields.split(",") if f.strip()),
        per_field=args.per_field,
        per_variant=args.per_variant,
        max_seeds=args.max_seeds,
        per_hypernym=args.per_hypernym,
        per_mechanism=args.per_mechanism,
        neighbor_top=args.neighbor_top,
        per_neighbor=args.per_neighbor,
    )
    result = expander.expand(args.construct)
    payload = {
        "construct": result.construct,
        "counts": result.counts(),
        "skipped": result.skipped,
        "terms": [
            {
                "text": t.text,
                "kind": t.kind,
                "score": t.score,
                "works": t.works,
                "found_via": list(t.found_via),
            }
            for t in result.terms
        ],
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for t in result.terms:
            score = "" if t.score is None else f"{t.score:.3f}"
            works = "" if t.works is None else str(t.works)
            print(f"{t.kind:<9} {score:>6} {works:>9}  {t.text}")
    counts = result.counts()
    parts = ", ".join(f"{n} {kind}" for kind, n in counts.items())
    print(
        f"{len(result.terms)} terms: {parts}"
        + (f"; {len(result.skipped)} step(s) skipped" if result.skipped else ""),
        file=sys.stderr,
    )
    for problem in getattr(expander.synonyms, "problems", []):
        print(f"  the model step had a problem: {problem}", file=sys.stderr)
    for what, why in result.skipped.items():
        print(f"  skipped {what!r}: {why}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")  # before the parser reads its defaults from the environment
    args = build_parser().parse_args(argv)
    if args.command in ("run", "serve"):
        corpus_names(args)  # a bad --corpus fails now, before any work starts
    if args.command == "show":
        out = args.out or args.file.with_suffix(".html")
        try:
            out.write_text(page_from_json(load_known(args.file)), encoding="utf-8")
        except ValueError as err:
            print(f"error: {args.file}: {err}", file=sys.stderr)
            return 1
        print(f"wrote {out.resolve()}")
        return 0
    if args.command == "review":
        return review(args.input, args.format)
    if args.command == "run" and args.replay and args.no_cache:
        print("error: --replay answers from the saved replies, which --no-cache turns off", file=sys.stderr)
        return 1
    if args.command == "run" and args.json and args.review == "text":
        print("error: --review text prints to the terminal, which --json keeps for the JSON", file=sys.stderr)
        return 1
    if args.command == "run":
        args.status = StatusLine()
        args.runlog = runlog = RunLog(
            args.log or args.out.with_name(f"{args.out.stem}.log.jsonl"), echo=args.status
        )
        runlog.event("run_started", **describe_run(args))
        warmed = None if args.no_cache else seed_demo_cache(args.cache_dir or cache_dir(), (args.iv, args.dv))
        if warmed:  # the built-in example question starts with its saved model answers (and searches)
            runlog.event("prewarmed_cache", state=warmed)
            if any(
                n not in ("file", "demo") for n in corpus_names(args)
            ):  # searched live: say they are saved
                print(f"{PREWARMED_NOTE}\n", file=sys.stderr)
    if args.command == "setup":
        return setup(args)
    if args.command == "flush":
        return flush_saved(args)
    if args.command == "demo":
        return demo(args)
    if args.command == "expand":
        return expand(args)
    if args.command == "install-skill":
        return install_skill(args)
    if args.command == "audit":
        trail = AuditTrail.load(args.file)
        if args.json:
            print(json.dumps({"file": str(args.file), **trail.to_dict()}, indent=2))
        else:
            print(trail.to_markdown(), end="")
        return 0 if trail.intact and trail.complete else 1
    backend = build_backend(args)
    env, proposer = backend.env, backend.proposer
    if args.command == "serve":
        manager = RunManager(
            env,
            proposer,
            args.max_concurrent,
            audit_dir=args.audit_dir,
            policy_factory=(lambda run: deterministic_policy(args, backend.client))
            if args.policy == "deterministic"
            else None,
            default_steps=RUN_STEPS if args.policy == "deterministic" else 40,
        )
        server = make_server(manager, args.host, args.port)
        print(f"ConfoundSearch at http://{args.host}:{server.server_port}", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            wait_for_writes()  # saved searches still waiting to be written land before the program ends
        return 0
    policy = build_policy(args, backend)
    max_steps = args.max_steps or (RUN_STEPS if args.policy == "deterministic" else 40)
    audit_path = args.audit or args.out.with_name(f"{args.out.stem}.audit.jsonl")
    for folder in {args.out.parent, audit_path.parent}:
        folder.mkdir(
            parents=True, exist_ok=True
        )  # `--out results/org.html` makes results/ if it is not there
    failed: str | None = None
    stopped: str | None = None
    with AuditLog(audit_path) as log:
        steps: list[Step] = []
        print(f"{EPIGRAPH}\n", file=sys.stderr)

        def show(step: Step) -> None:
            steps.append(step)
            graph = step.next_state.graph
            runlog.event(
                "step",
                n=len(steps),
                action=describe(step.action),
                reward=round(step.reward, 4),
                phi=round(phi(graph), 4),
                connected=len(graph.connected()),
            )
            if not args.status.live:  # a terminal shows the status line instead of a line per step
                print(progress(len(steps), step), file=sys.stderr, flush=True)

        args.status.total = max_steps
        args.status.start()
        try:
            trajectory = run_episode(
                env, policy, proposer, args.iv, args.dv, max_steps, on_step=show, audit=log
            )
        except BudgetReached as err:  # the money allowed has been spent: a planned stop, with partial results
            stopped = str(err)
            trajectory = Trajectory(env.initial_state(args.iv, args.dv), steps, audit=log)
            runlog.event("budget_reached", message=stopped, steps=len(steps))
        except (
            RemoteError,
            ReplayMiss,
        ) as err:  # a search or the model refused: keep what the run had so far
            failed = f"{type(err).__name__}: {err}"
            trajectory = Trajectory(env.initial_state(args.iv, args.dv), steps, audit=log)
            runlog.event("run_failed", error=failed, steps=len(steps))
        except BaseException as err:  # a bug, or Ctrl-C: say so in the log, then let it through
            runlog.event("crashed", error=f"{type(err).__name__}: {err}", traceback=traceback.format_exc())
            runlog.close()
            raise
        finally:
            args.status.stop()
    spent = float(getattr(getattr(backend.client, "client", None), "dollars", 0.0))
    if failed is None and stopped is None:
        runlog.event("run_finished", steps=len(steps), score=round(trajectory.score, 4), cost=round(spent, 4))
    runlog.close()
    wait_for_writes()  # the searches this run made are on disk before it says it is done
    share = to_share_data(trajectory.final)
    args.out.write_text(standalone_html(share), encoding="utf-8")
    if args.json:
        result = run_result(trajectory, args.out, audit_path)
        if failed:
            result["error"] = failed
        if stopped:
            result["stopped"] = stopped
        print(json.dumps(result, indent=2))
    else:
        graph = trajectory.final.graph
        print(
            f"{len(trajectory.steps)} steps, {len(graph.connected())} connected, phi={trajectory.score:.3f}"
        )
        print(where_it_went(args.out, audit_path, trajectory.audit_trail().intact, runlog.path))
        if spent:
            print(f"language model cost, estimated from tokens: about ${spent:.2f}")
        cached = find_cached(backend.client)
        if cached is not None and cached.hits:
            calls = cached.hits + cached.misses
            print(f"{cached.hits} of {calls} model calls were answered from saved replies, at no cost")
    if failed:
        print(
            f"error: the run stopped early: {failed}\nWhat it had found so far is in {args.out}; "
            f"what happened is in {runlog.path}.",
            file=sys.stderr,
        )
    if stopped:
        print(
            f"stopped at the budget: {stopped}. What it had found so far is in {args.out}.", file=sys.stderr
        )
    if args.review == "text":
        print(review_text(share), end="")
    elif args.review:
        open_page(args.out)
    return 1 if failed else 0
