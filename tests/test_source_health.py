"""MultiCorpus remembers each source's last success and leaves a failed one alone for a while."""

import pytest

from confoundsearch import CorpusError, InMemoryCorpus, Passage
from confoundsearch.data.multi import MultiCorpus
from confoundsearch.errors import RemoteError

GOOD = [Passage("g1", "firm size and growth", "Good paper", "")]


class Flaky:
    """A source that fails while `down` is set, and counts how often it was asked."""

    def __init__(self, down=False, error=None):
        self.down, self.asked, self.error = down, [], error or RemoteError("timed out waiting for x")

    def search(self, query, k):
        self.asked.append(query)
        if self.down:
            raise self.error
        return [Passage("f1", "flaky text", "Flaky paper", "")]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def setup(**kwargs):
    clock, events = Clock(), []
    flaky = Flaky(down=True)
    multi = MultiCorpus(
        {"good": InMemoryCorpus(GOOD), "flaky": flaky},
        lambda kind, **info: events.append((kind, info)),
        clock,
        **kwargs,
    )
    return multi, flaky, clock, events


def kinds(events):
    return [kind for kind, _ in events]


def test_a_failed_source_is_left_alone_for_the_cooldown_then_asked_again():
    multi, flaky, clock, events = setup()
    multi.search("firm size", 5)
    assert flaky.asked == ["firm size"]  # asked, and it failed
    clock.now += 4
    multi.search("firm size growth", 5)
    assert flaky.asked == ["firm size"]  # resting: not asked again, so no timeout paid
    assert ("source_skipped", {"source": "flaky", "retry_in": 1.0}) in events
    clock.now += 2  # the 5 seconds are over
    multi.search("firm size again", 5)
    assert flaky.asked == ["firm size", "firm size again"]


def test_the_cooldown_doubles_with_each_failure_in_a_row_up_to_a_limit():
    multi, _, clock, _ = setup(cooldown=(5.0, 12.0))
    waits = []
    for _ in range(4):
        multi.search("firm size", 5)
        waits.append(multi.health()["flaky"]["retry_in"])
        clock.now += waits[-1]
    assert waits == [5.0, 10.0, 12.0, 12.0]  # 5, 10, then the 12 second ceiling
    assert multi.health()["flaky"]["failures"] == 4


def test_one_success_clears_the_record_and_says_so():
    multi, flaky, clock, events = setup()
    multi.search("firm size", 5)
    clock.now += 6
    flaky.down = False
    found = multi.search("firm size back", 5)
    assert {p.title for p in found} == {"Good paper", "Flaky paper"}
    assert ("source_recovered", {"source": "flaky", "after_failures": 1}) in events
    state = multi.health()["flaky"]
    assert state["ok"] and state["failures"] == 0 and state["retry_in"] == 0.0 and state["last_error"] == ""
    multi.search("quiet", 5)
    assert kinds(events).count("source_recovered") == 1  # said once, not on every later success


def test_it_remembers_each_sources_last_successful_query():
    multi, _, clock, _ = setup()
    multi.search("first query", 5)
    clock.now += 30
    multi.search("second query", 5)
    health = multi.health()
    assert health["good"]["last_ok_query"] == "second query" and health["good"]["seconds_since_ok"] == 0.0
    assert health["flaky"]["last_ok_query"] is None and health["flaky"]["seconds_since_ok"] is None
    assert "timed out" in health["flaky"]["last_error"]
    clock.now += 7.5
    assert multi.health()["good"]["seconds_since_ok"] == 7.5  # and how long ago that was


def test_when_every_source_is_cooling_down_they_are_all_asked_anyway():
    clock, events = Clock(), []
    a, b = Flaky(down=True), Flaky(down=True)
    multi = MultiCorpus({"a": a, "b": b}, lambda kind, **info: events.append(kind), clock)
    with pytest.raises(CorpusError, match="every source failed"):
        multi.search("x", 5)
    a.down = False  # a recovers while both are still resting
    assert [p.title for p in multi.search("x again", 5)] == ["Flaky paper"]
    assert a.asked == ["x", "x again"] and b.asked == ["x", "x again"]
    assert "source_skipped" not in events  # nothing was skipped: everything was asked


def test_if_the_awake_sources_all_fail_the_resting_ones_are_the_last_hope():
    clock = Clock()
    awake, resting = Flaky(down=True), Flaky(down=True)
    multi = MultiCorpus({"awake": awake, "resting": resting}, None, clock)
    with pytest.raises(CorpusError):
        multi.search("first", 5)  # both fail and begin resting
    awake.down = resting.down = False
    clock.now += 6
    awake.down = True  # awake's cooldown is over and it fails again; 'resting' was never skipped here
    multi.state["resting"].skip_until = clock.now + 100  # but now pretend 'resting' is still resting
    found = multi.search("second", 5)
    assert [p.title for p in found] == ["Flaky paper"]
    assert resting.asked == ["first", "second"]  # asked as the last resort


def test_a_timeout_counts_as_a_failure_like_any_other():
    multi, flaky, _, _ = setup()
    flaky.error = RemoteError("timed out waiting for https://x.test/")
    assert [p.title for p in multi.search("firm size", 5)] == ["Good paper"]
    assert multi.health()["flaky"]["failures"] == 1


def test_health_before_any_search():
    multi, *_ = setup()
    assert multi.health()["good"] == {
        "ok": True,
        "failures": 0,
        "last_error": "",
        "last_ok_query": None,
        "seconds_since_ok": None,
        "retry_in": 0.0,
    }


def test_the_search_event_names_the_sources_that_were_skipped():
    multi, _, clock, events = setup()
    multi.search("firm size", 5)
    clock.now += 1
    multi.search("firm growth", 5)
    search = [info for kind, info in events if kind == "search"][-1]
    assert search["skipped"] == ["flaky"] and search["answered"] == {"good": 1} and search["failed"] == []
