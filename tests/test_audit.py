from __future__ import annotations

import itertools
import json
import threading

import pytest

import confoundsearch
from confoundsearch import (
    AuditedClient,
    AuditLog,
    AuditTrail,
    Environment,
    EpsilonGreedy,
    InMemoryCorpus,
    LLMError,
    Softmax,
    verify,
)
from confoundsearch.logic.audit import GENESIS, Event, action_dict, current, passage_dict, recording
from confoundsearch.logic.episode import run_episode
from tests.conftest import DV, IV, P1, P2, P3, StubJudge
from tests.test_episode import SCRIPT, Scripted


def ticking():
    counter = itertools.count()
    return lambda: f"2000-01-01T00:00:{next(counter):02d}.000+00:00"


def scripted_run(env, script=SCRIPT, **kwargs):
    agent = Scripted(*script)
    return run_episode(env, agent, agent, IV, DV, **kwargs)


def kinds(trajectory):
    return [e.kind for e in trajectory.audit.events]


class TestChain:
    def log(self, n=4):
        log = AuditLog(clock=ticking())
        for i in range(n):
            log.step = i
            log.record("note", i=i, text="é")
        return log

    def test_a_fresh_log_verifies_and_has_the_genesis_head(self):
        log = AuditLog()
        assert log.events == () and log.head == GENESIS and verify(log.events)

    def test_events_are_numbered_stamped_and_chained(self):
        log = self.log()
        assert [e.seq for e in log.events] == [0, 1, 2, 3]
        assert [e.step for e in log.events] == [0, 1, 2, 3]
        assert log.events[0].time == "2000-01-01T00:00:00.000+00:00"
        assert len({e.hash for e in log.events}) == 4 and log.head == log.events[-1].hash
        assert verify(log.events)

    def test_the_default_clock_gives_utc_timestamps(self):
        assert AuditLog().record("note").time.endswith("+00:00")

    def test_identical_content_gives_identical_hashes(self):
        assert [e.hash for e in self.log().events] == [e.hash for e in self.log().events]

    def test_verify_accepts_dicts_read_back_from_json(self):
        rows = [json.loads(line) for line in self.log().to_jsonl().splitlines()]
        assert verify(rows)

    @pytest.mark.parametrize(
        "tamper",
        [
            lambda rows: rows[1]["data"].update(i=99),  # edit an event
            lambda rows: rows[2].update(kind="other"),
            lambda rows: rows[0].update(time="1999-01-01T00:00:00.000+00:00"),
            lambda rows: rows[1].update(hash="0" * 64),
            lambda rows: rows[3].update(seq=7),
            lambda rows: rows.pop(1),  # delete from the middle
            lambda rows: rows.pop(0),  # delete the first
            lambda rows: rows.reverse(),
            lambda rows: rows.insert(2, dict(rows[1])),  # duplicate an event
            lambda rows: rows[2].pop("hash"),
        ],
    )
    def test_any_tampering_breaks_the_chain(self, tamper):
        rows = [json.loads(line) for line in self.log().to_jsonl().splitlines()]
        tamper(rows)
        assert not verify(rows)

    def test_cutting_off_the_end_is_not_detected_by_the_chain_but_shows_as_incomplete(self):
        rows = [json.loads(line) for line in self.log().to_jsonl().splitlines()][:2]
        assert verify(
            rows
        )  # a prefix is a valid chain; the head hash (and the end-of-run event) tell the story

    def test_data_that_is_not_json_is_refused_and_leaves_the_log_untouched(self):
        log = self.log(2)
        with pytest.raises(TypeError):
            log.record("bad", value=frozenset({1}))
        assert len(log.events) == 2 and verify(log.events)

    def test_many_threads_recording_at_once_keep_one_unbroken_chain(self):
        log = AuditLog()
        workers = [threading.Thread(target=lambda: [log.record("n") for _ in range(50)]) for _ in range(8)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert len(log.events) == 400 and verify(log.events)
        assert [e.seq for e in log.events] == list(range(400))


class TestFiles:
    def test_every_event_reaches_the_file_as_it_happens(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path, clock=ticking())
        log.record("first")
        assert len(path.read_text().splitlines()) == 1  # on disk before the log is closed
        log.record("second")
        log.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["kind"] for r in rows] == ["first", "second"] and verify(rows)

    def test_close_is_safe_to_repeat_and_the_context_manager_closes(self, tmp_path):
        with AuditLog(tmp_path / "a.jsonl") as log:
            log.record("x")
        log.close()
        assert log._sink is None and log.path == tmp_path / "a.jsonl"

    def test_save_writes_the_whole_log(self, tmp_path):
        log = AuditLog(clock=ticking())
        log.record("a")
        log.record("b")
        saved = log.save(tmp_path / "out.jsonl")
        assert saved.read_text(encoding="utf-8") == log.to_jsonl()
        assert AuditTrail.load(saved).intact

    def test_an_empty_log_saves_an_empty_file(self, tmp_path):
        assert AuditLog().save(tmp_path / "e.jsonl").read_text() == ""

    def test_load_skips_blank_lines(self, tmp_path):
        log = AuditLog()
        log.record("a")
        path = tmp_path / "x.jsonl"
        path.write_text("\n" + log.to_jsonl() + "\n\n")
        assert len(AuditTrail.load(path).events) == 1


class TestClient:
    class Model:
        model = "test-model"

        def __init__(self, reply="hello", error=None):
            self.reply, self.error = reply, error

        def chat(self, messages):
            if self.error:
                raise self.error
            return self.reply

    def test_records_prompt_reply_and_model_to_an_explicit_log(self):
        log = AuditLog()
        messages = [{"role": "user", "content": "hi"}]
        assert AuditedClient(self.Model(), log).chat(messages) == "hello"
        (event,) = log.events
        assert event.kind == "model_call" and event.data == {
            "model": "test-model", "messages": messages, "reply": "hello",
        }  # fmt: skip

    def test_the_recorded_messages_are_a_copy(self):
        log = AuditLog()
        messages = [{"role": "user", "content": "hi"}]
        AuditedClient(self.Model(), log).chat(messages)
        messages[0]["content"] = "changed later"
        assert log.events[0].data["messages"][0]["content"] == "hi"

    def test_a_client_without_a_model_attribute_records_null(self):
        class Bare:
            def chat(self, messages):
                return "x"

        log = AuditLog()
        AuditedClient(Bare(), log).chat([])
        assert log.events[0].data["model"] is None

    def test_records_to_the_log_of_the_running_episode(self, env):
        client = AuditedClient(self.Model())
        assert current() is None
        with recording(AuditLog()) as log:
            assert current() is log
            client.chat([{"role": "user", "content": "inside"}])
        assert current() is None and log.events[0].data["messages"][0]["content"] == "inside"

    def test_outside_a_run_it_passes_calls_through_and_records_nothing(self):
        assert AuditedClient(self.Model("fine")).chat([]) == "fine"

    def test_failures_are_recorded_and_still_raised(self):
        log = AuditLog()
        with pytest.raises(LLMError):
            AuditedClient(self.Model(error=LLMError("HTTP 500")), log).chat(
                [{"role": "user", "content": "q"}]
            )
        (event,) = log.events
        assert event.kind == "model_error" and "LLMError: HTTP 500" in event.data["error"]

    def test_a_failure_outside_a_run_is_simply_raised(self):
        with pytest.raises(LLMError):
            AuditedClient(self.Model(error=LLMError("HTTP 500"))).chat([])

    def test_nested_recording_restores_the_outer_log(self):
        with recording(AuditLog()) as outer:
            with recording(AuditLog()) as inner:
                assert current() is inner
            assert current() is outer


class TestRun:
    def test_the_log_is_always_kept_and_attached_to_the_trajectory(self, env):
        trajectory = scripted_run(env)
        assert isinstance(trajectory.audit, AuditLog) and trajectory.audit.events
        assert isinstance(trajectory.audit_trail(), AuditTrail)

    def test_a_log_you_pass_in_is_the_one_used(self, env):
        mine = AuditLog()
        assert scripted_run(env, audit=mine).audit is mine and mine.events

    def test_it_can_stream_to_a_file_while_running(self, env, tmp_path):
        path = tmp_path / "live.jsonl"
        with AuditLog(path) as log:
            trajectory = scripted_run(env, audit=log)
        assert AuditTrail.load(path).events == trajectory.audit.events

    def test_events_come_in_order_from_start_to_finish(self, env):
        order = kinds(scripted_run(env))
        assert order[0] == "run_started" and order[-1] == "run_finished"
        assert order.count("action") == len(SCRIPT)
        assert order.count("retrieved") == 1 and order.count("judged") == 2
        assert order[order.index("retrieved") - 1] == "action"  # each effect follows its action

    def test_the_log_is_intact_and_complete(self, env):
        trail = scripted_run(env).audit_trail()
        assert trail.intact and trail.complete

    def test_run_started_records_the_question_and_the_policy(self, env):
        policy = EpsilonGreedy(lambda s, a: 0.0, epsilon=0.25, explore_stop=False)

        class Nothing:
            def propose(self, state):
                return []

        trajectory = run_episode(env, policy, Nothing(), IV, DV, max_steps=7)
        started = trajectory.audit.events[0].data
        assert started["independent_variable"] == IV and started["dependent_variable"] == DV
        assert started["max_steps"] == 7 and started["policy"] == "EpsilonGreedy"
        assert started["policy_settings"] == {"epsilon": 0.25, "explore_stop": False}
        assert started["package_version"] == confoundsearch.__version__

    def test_softmax_settings_are_recorded_too(self, env):
        class Nothing:
            def propose(self, state):
                return []

        trajectory = run_episode(env, Softmax(lambda s, a: 0.0, temperature=2.0), Nothing(), IV, DV)
        assert trajectory.audit.events[0].data["policy_settings"] == {"temperature": 2.0}

    def test_a_policy_with_no_settings_records_none(self, env):
        agent = Scripted(*SCRIPT)
        assert run_episode(env, agent, agent, IV, DV).audit.events[0].data["policy_settings"] == {}

    def test_actions_carry_their_reward_and_the_score_before_and_after(self, env):
        trajectory = scripted_run(env)
        actions = [e.data for e in trajectory.audit.events if e.kind == "action"]
        assert actions[0]["action"] == {"type": "AddConfound", "label": "firm size", "strings": ["firm size"]}
        assert actions[2]["action"] == {"type": "AddEdge", "confound": "firm size", "target": "Y"}
        assert actions[2]["reward"] == pytest.approx(1.0) and actions[2]["score_before"] == 0.0
        assert actions[-1]["action"] == {"type": "Stop"}
        for entry, step in zip(actions, trajectory.steps):
            assert entry["reward"] == step.reward
            assert entry["score_after"] - entry["score_before"] == pytest.approx(step.reward)

    def test_a_search_logs_the_query_and_the_full_text_of_what_it_found(self, env):
        trajectory = scripted_run(env)
        (retrieved,) = [e for e in trajectory.audit.events if e.kind == "retrieved"]
        assert retrieved.data["query"] == "larger firms" and retrieved.data["requested"] == 5
        assert retrieved.data["new_passages"] == [passage_dict(P1), passage_dict(P2)]
        assert retrieved.data["passages_seen_so_far"] == 2

    def test_a_second_search_logs_only_the_new_passages(self, env):
        from confoundsearch import Retrieve

        agent = Scripted(Retrieve("larger firms"), Retrieve("larger firms"), Retrieve("head count"))
        trajectory = run_episode(env, agent, agent, IV, DV)
        found = [e.data["new_passages"] for e in trajectory.audit.events if e.kind == "retrieved"]
        assert [len(f) for f in found] == [2, 0, 1] and found[2][0]["id"] == "p3"

    def test_every_citation_is_logged_with_the_claim_and_the_verdict(self, env):
        trajectory = scripted_run(env)
        judged = [e.data for e in trajectory.audit.events if e.kind == "judged"]
        assert [(j["claim"], j["verdict"], j["passage"]["id"]) for j in judged] == [
            (f"firm size affects {IV}", "supported", "p1"),
            (f"firm size affects {DV}", "supported", "p2"),
        ]

    def test_a_rejected_citation_is_logged_as_rejected(self, corpus):
        env = Environment(corpus, StubJudge({"p1"}))
        judged = [e.data["verdict"] for e in scripted_run(env).audit.events if e.kind == "judged"]
        assert judged == ["supported", "rejected"]

    def test_steps_are_numbered_and_the_final_event_has_the_step_count(self, env):
        events = scripted_run(env).audit.events
        assert events[0].step == 0 and [e.step for e in events if e.kind == "action"] == list(range(1, 8))
        assert events[-1].step == 7 and events[-1].data["steps"] == 7

    def test_run_finished_summarises_the_graph_and_what_backs_each_link(self, env):
        final = scripted_run(env).audit.events[-1].data
        assert final["reason"] == "stop_action" and final["final_score"] > 1.0
        assert final["passages_observed"] == 2 and final["passages_rejected"] == 0
        (confound,) = final["graph"]["confounds"]
        assert confound == {
            "label": "firm size", "strings": ["firm size"], "connected": True,
            "supporting_passages": {"X": ["p1"], "Y": ["p2"]},
        }  # fmt: skip

    def test_why_a_run_ended_is_recorded(self, env):
        class Forever:
            def propose(self, state):
                from confoundsearch import Retrieve

                return [Retrieve("larger firms")]

            def select(self, state, actions):
                return actions[0]

        class NoMoves:
            def propose(self, state):
                return []

        budget = run_episode(env, Forever(), Forever(), IV, DV, max_steps=3)
        none = run_episode(env, Forever(), NoMoves(), IV, DV)
        asked = run_episode(env, Forever(), Forever(), IV, DV, should_stop=lambda: True)
        stop = scripted_run(env)
        reasons = [t.audit.events[-1].data["reason"] for t in (budget, none, asked, stop)]
        assert reasons == ["step_budget", "no_actions", "stopped_by_user", "stop_action"]

    def test_a_failure_is_logged_and_still_raised(self, env):
        class Broken:
            def propose(self, state):
                return [object()]

            def select(self, state, actions):
                return actions[0]

        log = AuditLog()
        with pytest.raises(confoundsearch.InvalidAction):
            run_episode(env, Broken(), Broken(), IV, DV, audit=log)
        last = log.events[-1]
        assert last.kind == "run_failed" and "InvalidAction" in last.data["error"] and verify(log.events)
        trail = AuditTrail.from_log(log)
        assert trail.complete and trail.intact  # it ended, and said why

    def test_model_calls_land_in_the_log_between_the_steps_they_belong_to(self, env):
        class Model:
            model = "m"

            def chat(self, messages):
                return "yes"

        client = AuditedClient(Model())

        class Asks(Scripted):
            def propose(self, state):
                client.chat([{"role": "user", "content": "what next?"}])
                return super().propose(state)

        agent = Asks(*SCRIPT)
        trajectory = run_episode(env, agent, agent, IV, DV)
        events = trajectory.audit.events
        calls = [e for e in events if e.kind == "model_call"]
        assert len(calls) == len(SCRIPT) and all(e.data["reply"] == "yes" for e in calls)
        for call in calls:
            action = next(e for e in events if e.kind == "action" and e.step == call.step)
            assert call.seq < action.seq  # the question was asked before the move it led to

    def test_the_same_run_gives_the_same_log_every_time(self, corpus):
        def once():
            env = Environment(corpus, StubJudge())
            agent = Scripted(*SCRIPT)
            return run_episode(env, agent, agent, IV, DV, audit=AuditLog(clock=ticking())).audit

        a, b = once(), once()
        assert a.to_jsonl() == b.to_jsonl() and a.head == b.head


class TestTrail:
    def trail(self, env, script=SCRIPT):
        return scripted_run(env, script).audit_trail()

    def test_evidence_rows_say_where_each_passage_came_from(self, env):
        rows = self.trail(env).evidence
        assert [(r.confound, r.target, r.passage_id, r.verdict) for r in rows] == [
            ("firm size", "X", "p1", "supported"),
            ("firm size", "Y", "p2", "supported"),
        ]
        first = rows[0]
        assert (
            first.retrieved_by == "larger firms"
            and first.retrieved_at_step == 4
            and first.judged_at_step == 5
        )
        assert first.title == "" and first.url == ""

    def test_supporting_filters_by_target_and_leaves_out_rejections(self, corpus):
        trail = self.trail(Environment(corpus, StubJudge({"p1"})))
        assert [r.passage_id for r in trail.supporting("firm size")] == ["p1"]
        assert [r.passage_id for r in trail.supporting("firm size", "X")] == ["p1"]
        assert trail.supporting("firm size", "Y") == () and trail.supporting("ghost") == ()
        assert {r.verdict for r in trail.evidence} == {"supported", "rejected"}

    def test_a_passage_judged_without_a_recorded_search_has_no_provenance(self):
        log = AuditLog()
        passage = {"id": "z", "title": "T", "url": "u", "text": "t"}
        log.record(
            "judged", claim="c affects x", confound="c", target="X", verdict="supported", passage=passage
        )
        (row,) = AuditTrail.from_log(log).evidence
        assert row.retrieved_by == "" and row.retrieved_at_step == 0

    def test_of_kind(self, env):
        trail = self.trail(env)
        assert len(trail.of_kind("action")) == len(SCRIPT) and trail.of_kind("nothing") == ()

    def test_a_saved_log_reads_back_the_same(self, env, tmp_path):
        trajectory = scripted_run(env)
        loaded = AuditTrail.load(trajectory.audit.save(tmp_path / "a.jsonl"))
        assert (
            loaded.events == trajectory.audit.events and loaded.evidence == trajectory.audit_trail().evidence
        )
        assert loaded.intact and loaded.complete

    def test_a_tampered_file_is_caught_on_load(self, env, tmp_path):
        path = scripted_run(env).audit.save(tmp_path / "a.jsonl")
        lines = path.read_text().splitlines()
        row = json.loads(lines[5])
        row["data"]["reward"] = 99.0
        lines[5] = json.dumps(row)
        path.write_text("\n".join(lines) + "\n")
        assert not AuditTrail.load(path).intact

    def test_a_truncated_file_is_incomplete_though_its_chain_holds(self, env, tmp_path):
        path = scripted_run(env).audit.save(tmp_path / "a.jsonl")
        path.write_text("\n".join(path.read_text().splitlines()[:6]) + "\n")
        trail = AuditTrail.load(path)
        assert trail.intact and not trail.complete

    def test_an_empty_trail_is_intact_but_not_complete(self):
        trail = AuditTrail(())
        assert trail.intact and not trail.complete and trail.evidence == ()

    def test_markdown_lays_out_the_evidence_and_says_whether_the_log_can_be_trusted(self, env):
        text = self.trail(env).to_markdown()
        assert text.startswith("# Audit trail") and text.endswith("\n")
        assert f"does **{IV}** affect **{DV}**" in text and "stop action" in text
        assert "hash chain intact" in text and "complete" in text
        assert f"**firm size → {IV}**: 1 supported, 0 rejected" in text
        assert '`p1`): found by "larger firms" at step 4, accepted at step 5' in text

    def test_markdown_includes_titles_urls_and_counts_rejections(self):
        from confoundsearch import Passage, Retrieve

        passages = [
            Passage("a", "firm size and IT", "A title", "https://x/a"),
            Passage("b", "firm size and profits"),
        ]
        env = Environment(InMemoryCorpus(passages), StubJudge({"a"}))
        script = [
            *SCRIPT[:3], Retrieve("firm size"),
            confoundsearch.AttachPassage("firm size", "X", "a"),
            confoundsearch.AttachPassage("firm size", "X", "b"),
        ]  # fmt: skip
        text = self.trail(env, script).to_markdown()
        assert "A title (`a`) <https://x/a>" in text and ": 1 supported, 1 rejected" in text
        assert "budget" not in text and "no end-of-run" not in text

    def test_markdown_flags_a_broken_or_unfinished_log(self, env, tmp_path):
        path = scripted_run(env).audit.save(tmp_path / "a.jsonl")
        lines = path.read_text().splitlines()
        row = json.loads(lines[2])
        row["step"] = 42
        lines[2] = json.dumps(row)
        path.write_text("\n".join(lines[:-1]) + "\n")
        text = AuditTrail.load(path).to_markdown()
        assert "hash chain BROKEN" in text and "INCOMPLETE" in text

    def test_markdown_of_an_empty_trail(self):
        assert "0 events" in AuditTrail(()).to_markdown()

    def test_markdown_without_a_start_event_still_names_the_targets(self):
        log = AuditLog()
        passage = {"id": "z", "title": "", "url": "", "text": "t"}
        log.record("judged", claim="c", confound="c", target="Y", verdict="supported", passage=passage)
        text = AuditTrail.from_log(log).to_markdown()
        assert "**c → Y**: 1 supported" in text


class TestHelpers:
    def test_action_dict_names_the_type_and_sorts_sets(self):
        from confoundsearch import AddConfound, Merge, Stop

        assert action_dict(Stop()) == {"type": "Stop"}
        assert action_dict(AddConfound("a", frozenset({"z", "b"}))) == {
            "type": "AddConfound",
            "label": "a",
            "strings": ["b", "z"],
        }
        assert action_dict(Merge("a", "b")) == {"type": "Merge", "a": "a", "b": "b", "label": None}

    def test_passage_dict(self):
        assert passage_dict(P3) == {"id": "p3", "title": "", "url": "", "text": P3.text}

    def test_event_round_trips_through_a_dict(self):
        event = AuditLog(clock=ticking()).record("x", a=1)
        assert Event(**event.to_dict()) == event
