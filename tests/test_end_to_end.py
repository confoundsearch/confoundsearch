"""The whole agent against a fake OpenAI-compatible server, over real HTTP."""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from confoundsearch import (
    Environment,
    EpsilonGreedy,
    InMemoryCorpus,
    LLMJudge,
    LLMProposer,
    LookaheadQ,
    OpenAICompatibleClient,
    Passage,
    Proposer,
    run_episode,
)

CORPUS = InMemoryCorpus(
    [
        Passage("a", "firm size drives both IT investment and firm performance"),
        Passage("b", "larger firms invest more in IT"),
        Passage("c", "the weather was pleasant"),
    ]
)

PROPOSAL = {
    "queries": ["firm size IT investment"],
    "confounds": [{"label": "firm size", "strings": ["firm size", "head count"]}],
}


def respond(prompt: str) -> str:
    if "Reply with JSON only" in prompt:
        return json.dumps(PROPOSAL)
    return "yes" if "firm size" in prompt else "no"


@pytest.fixture
def endpoint():
    prompts: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            prompts.append(prompt)
            reply = {"choices": [{"message": {"role": "assistant", "content": respond(prompt)}}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", prompts
    httpd.shutdown()
    httpd.server_close()


def test_agent_discovers_a_supported_confound(endpoint):
    url, prompts = endpoint
    client = OpenAICompatibleClient("any-model", base_url=url)
    env = Environment(CORPUS, LLMJudge(client))
    proposer = Proposer(LLMProposer(client))
    policy = EpsilonGreedy(LookaheadQ(env.preview), epsilon=0.2, rng=random.Random(0))

    trajectory = run_episode(env, policy, proposer, "IT investment", "firm performance", max_steps=60)

    graph = trajectory.final.graph
    assert "firm size" in graph.labels
    assert graph.is_connected("firm size")
    assert graph.evidence("firm size") >= 1
    assert trajectory.score > 1.0
    assert trajectory.total_reward == pytest.approx(trajectory.score)
    assert any("Reply with JSON only" in p for p in prompts)  # the LLM proposed
    assert any("Does the passage" in p for p in prompts)  # the LLM judged evidence
