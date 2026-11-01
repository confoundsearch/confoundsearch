"""The spending cap: estimating what the model has cost, and stopping the run cleanly at the budget."""

from __future__ import annotations

import json

import pytest

from confoundsearch.errors import BudgetReached
from confoundsearch.interface import cli
from confoundsearch.model.cost import UNKNOWN_MODEL, Spend, price_of
from confoundsearch.model.llm import OpenAICompatibleClient
from tests.test_cli import COMMON, ScriptedClient, fake_backend


class TestSpend:
    def test_tokens_are_priced_by_model(self):
        spend = Spend("gpt-4.1-mini")
        spend.add({"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
        assert spend.dollars == pytest.approx(0.40 + 0.80)

    def test_an_openrouter_slug_and_the_longest_name_win(self):
        assert price_of("openai/gpt-4.1-mini") == price_of("gpt-4.1-mini") != price_of("gpt-4.1")
        assert price_of("GPT-4O-MINI-2024") == price_of("gpt-4o-mini")

    def test_an_unknown_model_is_assumed_expensive(self):
        assert price_of("qwen/qwen3-max") == UNKNOWN_MODEL

    def test_a_cost_the_provider_reports_is_used_instead(self):
        spend = Spend("qwen/qwen3-max")
        spend.add({"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.25})
        spend.add({"cost": 0.25})
        assert spend.dollars == 0.5

    def test_nonsense_usage_counts_as_nothing(self):
        spend = Spend("gpt-4.1")
        for usage in (None, "x", [], {"prompt_tokens": "many", "cost": None}):
            spend.add(usage)
        assert spend.dollars == 0.0 and (spend.tokens_in, spend.tokens_out) == (0, 0)


def reply(content="yes", usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode()


def test_the_client_counts_what_each_reply_says_it_used():
    replies = [reply(usage={"prompt_tokens": 100, "completion_tokens": 10}), reply()]
    client = OpenAICompatibleClient("gpt-4.1", api_key="k", transport=lambda *a: replies.pop(0))
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "again"}])  # a reply with no usage block adds nothing
    assert (client.spend.tokens_in, client.spend.tokens_out) == (100, 10)


class Costly:
    model = "gpt-4.1-mini"

    def __init__(self):
        self.spend = Spend(self.model)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        self.spend.add({"prompt_tokens": 1_000_000})  # $0.40 a call
        return "yes"


class TestAnnounced:
    def test_calls_go_through_until_the_budget_is_reached(self):
        inner, seen = Costly(), []
        client = cli.Announced(inner, lambda kind, **d: seen.append(kind), budget=0.50)
        client.chat([])  # $0.40: under
        client.chat([])  # $0.80 after this one
        with pytest.raises(BudgetReached, match=r"about \$0\.80, the budget of \$0\.50"):
            client.chat([])
        assert inner.calls == 2 and seen == ["asking_model", "asking_model"]

    def test_without_a_budget_nothing_stops_it(self):
        inner = Costly()
        client = cli.Announced(inner, None)
        for _ in range(5):
            client.chat([])
        assert inner.calls == 5 and client.dollars == pytest.approx(2.0)

    def test_a_client_that_does_not_count_costs_nothing(self):
        client = cli.Announced(ScriptedClient(), None, budget=0.01)
        assert client.chat([{"role": "user", "content": "x"}]) == "yes" and client.dollars == 0.0


class TestTheRun:
    def argv(self, tmp_path, *more):
        return [
            "run",
            *COMMON,
            "--corpus-file",
            "x",
            "--iv",
            "a",
            "--dv",
            "b",
            "--out",
            str(tmp_path / "p.html"),
            *more,
        ]

    def test_the_budget_is_a_run_option_and_goes_in_the_log(self):
        args = cli.build_parser().parse_args(
            self.argv(__import__("pathlib").Path("."), "--max-budget", "0.5")
        )
        assert args.max_budget == 0.5 and cli.describe_run(args)["max_budget"] == 0.5

    def test_reaching_it_stops_cleanly_keeping_the_work(self, tmp_path, monkeypatch, capsys, env):
        monkeypatch.setattr(cli, "build_backend", fake_backend(env))

        def stop(*a, **k):
            raise BudgetReached("the language model has cost about $0.60, the budget of $0.50")

        monkeypatch.setattr(cli, "run_episode", stop)
        code = cli.main(self.argv(tmp_path, "--max-budget", "0.5", "--json"))
        captured = capsys.readouterr()
        assert code == 0 and (tmp_path / "p.html").exists() and (tmp_path / "p.audit.jsonl").exists()
        assert "budget of $0.50" in json.loads(captured.out)["stopped"] and "error" not in json.loads(
            captured.out
        )
        assert "stopped at the budget" in captured.err
        events = [json.loads(line)["event"] for line in (tmp_path / "p.log.jsonl").read_text().splitlines()]
        assert events[-1] == "budget_reached"

    def test_a_finished_run_reports_what_it_cost(self, tmp_path, monkeypatch, capsys, env):
        inner = Costly()
        monkeypatch.setattr(
            cli,
            "build_backend",
            lambda args: cli.Backend(
                env, fake_backend(env)(args).proposer, cli.AuditedClient(cli.Announced(inner, None))
            ),
        )
        inner.spend.add({"prompt_tokens": 2_500_000})  # $1.00 before the run starts
        assert cli.main(self.argv(tmp_path, "--max-steps", "2")) == 0
        total = inner.spend.dollars
        assert total >= 1.0
        assert f"estimated from tokens: about ${total:.2f}" in capsys.readouterr().out
        last = json.loads((tmp_path / "p.log.jsonl").read_text().splitlines()[-1])
        assert last["event"] == "run_finished" and last["cost"] == round(total, 4)
