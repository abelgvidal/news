import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agregador import db
from agregador.render import Story
from agregador.summarize import (
    SUMMARY_MODEL, build_prompt, check_headline, needs_refresh, parse_response, select_sources,
    summarize_stories, summarize_story,
)

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def story(cid=1, n=3):
    sources = [{"name": f"M{i}", "title": f"Titular {i}", "snippet": "extracto " * 80,
                "url": "u", "published_at": NOW} for i in range(n)]
    return Story(cid, "Titular 0", "u", None, NOW, 1.0, sources)


class FakeClient:
    def __init__(self, payload=None, fail=False, payloads=None):
        self.calls, self.fail = 0, fail
        self.payloads = payloads or [payload or {"headline": "H", "summary": "S"}]
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("boom")
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        block = SimpleNamespace(type="text", text=json.dumps(payload))
        return SimpleNamespace(stop_reason="end_turn", content=[block])


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.migrate(c)
    c.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at) "
              "VALUES (1, 'm', x'00', 3, 'x', 'x')")
    return c


def test_prompt_truncates_snippets_and_lists_outlets():
    p = build_prompt(story())
    assert "[M0] Titular 0" in p and "[M2]" in p
    assert max(len(line) for line in p.splitlines()) < 400


def test_needs_refresh_only_after_meaningful_growth():
    assert needs_refresh(None, 5)
    assert not needs_refresh(10, 12)   # +2: below the minimum
    assert needs_refresh(10, 13)       # +3 and +30%
    assert not needs_refresh(30, 35)   # +5 but only 17%


def test_parse_response_rejects_empty():
    assert parse_response('{"headline": " A ", "summary": "B"}') == ("A", "B")
    with pytest.raises(ValueError):
        parse_response('{"headline": "", "summary": "B"}')


def test_summarize_stores_and_caches(conn):
    client = FakeClient()
    assert summarize_stories(conn, [story()], client) == 1
    row = conn.execute("SELECT * FROM summary").fetchone()
    assert (row["headline"], row["text"], row["model"]) == ("H", "S", SUMMARY_MODEL)
    assert client.kwargs["model"] == SUMMARY_MODEL
    assert summarize_stories(conn, [story()], client) == 0  # unchanged cluster: no new call
    assert client.calls == 1


def test_failure_keeps_previous_summary_and_does_not_raise(conn):
    summarize_stories(conn, [story(n=3)], FakeClient())
    bad = FakeClient(fail=True)
    summarize_stories(conn, [story(n=9)], bad)  # grew enough to refresh, but the call fails
    assert bad.calls == 1
    assert conn.execute("SELECT headline FROM summary").fetchone()["headline"] == "H"


def test_stops_after_consecutive_failures(conn):
    bad = FakeClient(fail=True)
    summarize_stories(conn, [story(cid=i) for i in range(1, 8)], bad)
    assert bad.calls == 3


def test_select_sources_gives_every_outlet_a_slot_before_repeats():
    # ordered most representative first: A has three articles, B and C one each
    srcs = [{"name": n, "title": t} for n, t in
            [("A", "a1"), ("A", "a2"), ("A", "a3"), ("B", "b1"), ("C", "c1")]]
    chosen = select_sources(srcs, limit=4)
    assert [s["title"] for s in chosen] == ["a1", "b1", "c1", "a2"]
    assert {s["name"] for s in select_sources(srcs, limit=3)} == {"A", "B", "C"}


def test_select_sources_keeps_everything_when_under_the_limit():
    srcs = [{"name": "A", "title": "a1"}, {"name": "B", "title": "b1"}]
    assert select_sources(srcs) == srcs


@pytest.mark.parametrize("headline", [
    "Trump se declara dispuesto a reunirse con Putin",
    "Vox estudia no presentarse al Senado",
    "Feijóo: «No voy a fallar»",
    'El ministro dice que bajará los impuestos',
    "¿Qué pasa con la dana?",
    "x" * 91,
])
def test_check_headline_flags_statements_speculation_quotes_and_length(headline):
    assert check_headline(headline)


@pytest.mark.parametrize("headline", [
    "Francia impone toques de queda parciales en París y otras ciudades",
    "Trump propone una cumbre con Putin y Xi en noviembre",
    "El Gobierno declara el estado de emergencia en tres regiones",  # 'declara' is a legal act, not a statement
    "Casación confirma el archivo de la causa contra Macri",
])
def test_check_headline_accepts_factual_headlines(headline):
    assert check_headline(headline) == []


def test_bad_headline_triggers_one_retry_that_names_the_broken_rule():
    client = FakeClient(payloads=[{"headline": "Trump se declara dispuesto", "summary": "S1"},
                                  {"headline": "Trump propone una cumbre", "summary": "S2"}])
    assert summarize_story(client, story()) == ("Trump propone una cumbre", "S2")
    assert client.calls == 2
    assert "se declara" in client.kwargs["messages"][0]["content"]


def test_good_headline_costs_a_single_call():
    client = FakeClient(payload={"headline": "Casación confirma el archivo", "summary": "S"})
    summarize_story(client, story())
    assert client.calls == 1


def test_still_bad_after_retry_is_kept_not_retried_forever():
    bad = {"headline": "El presidente dice que dimitirá", "summary": "S"}
    client = FakeClient(payloads=[bad, bad])
    assert summarize_story(client, story())[0] == bad["headline"]
    assert client.calls == 2
