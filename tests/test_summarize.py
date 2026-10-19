import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agregador import db
from agregador.render import Story
from agregador.summarize import (
    SUMMARY_TAG, UnverifiedSummary, VERIFY_SYSTEM, build_prompt, check_headline, check_lines,
    check_numbers, needs_refresh, parse_response, select_sources, summarize_stories, summarize_story,
)

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
GOOD = {"headline": "Francia impone toques de queda en París", "lines": ["Prohibida la circulación de 23.00 a 6.00.", "Balance: 2.400 detenidos."]}


def story(cid=1, n=3, snippet="extracto 2.400 detenidos de 23.00 a 6.00 " * 8):
    sources = [{"name": f"M{i}", "title": f"Titular {i}", "snippet": snippet, "url": "u", "published_at": NOW}
               for i in range(n)]
    return Story(cid, "Titular 0", "u", None, NOW, 1.0, sources)


class FakeClient:
    """Answers generation calls from `drafts` in order and verification calls from `verdicts`."""

    def __init__(self, drafts=None, verdicts=None, fail=False):
        self.drafts, self.verdicts, self.fail = drafts or [GOOD], verdicts or [[]], fail
        self.gen_calls = self.verify_calls = 0
        self.prompts = []
        self.messages = self

    def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        if kwargs["system"] == VERIFY_SYSTEM:
            payload = {"problemas": self.verdicts[min(self.verify_calls, len(self.verdicts) - 1)]}
            self.verify_calls += 1
        else:
            payload = self.drafts[min(self.gen_calls, len(self.drafts) - 1)]
            self.gen_calls += 1
            self.prompts.append(kwargs["messages"][0]["content"])
        block = SimpleNamespace(type="text", text=json.dumps(payload))
        return SimpleNamespace(stop_reason="end_turn", content=[block])


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.migrate(c)
    c.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at) "
              "VALUES (1, 'm', x'00', 3, 'x', 'x')")
    return c


# --- sources and prompt ---------------------------------------------------------------------

def test_prompt_truncates_snippets_and_lists_outlets():
    p = build_prompt(story())
    assert "[M0] Titular 0" in p and "[M2]" in p
    assert max(len(line) for line in p.splitlines()) < 400


def test_select_sources_gives_every_outlet_a_slot_before_repeats():
    srcs = [{"name": n, "title": t} for n, t in
            [("A", "a1"), ("A", "a2"), ("A", "a3"), ("B", "b1"), ("C", "c1")]]
    assert [s["title"] for s in select_sources(srcs, limit=4)] == ["a1", "b1", "c1", "a2"]
    assert {s["name"] for s in select_sources(srcs, limit=3)} == {"A", "B", "C"}


def test_select_sources_keeps_everything_when_under_the_limit():
    srcs = [{"name": "A", "title": "a1"}, {"name": "B", "title": "b1"}]
    assert select_sources(srcs) == srcs


def test_needs_refresh_only_after_meaningful_growth():
    assert needs_refresh(None, 5)
    assert not needs_refresh(10, 12)
    assert needs_refresh(10, 13)
    assert not needs_refresh(30, 35)


def test_parse_response_rejects_empty():
    assert parse_response('{"headline": " A ", "lines": [" B ", ""]}') == ("A", ["B"])
    with pytest.raises(ValueError):
        parse_response('{"headline": "", "lines": ["B"]}')


# --- deterministic checks -------------------------------------------------------------------

@pytest.mark.parametrize("headline", [
    "Trump se declara dispuesto a reunirse con Putin",
    "Vox estudia no presentarse al Senado",
    "Feijóo: «No voy a fallar»",
    "El ministro dice que bajará los impuestos",
    "Milei destaca en Neuquén el superávit energético",
    "Ravier atribuye a la incertidumbre la suba del riesgo país",
    "¿Qué pasa con la dana?",
    "x" * 81,
])
def test_check_headline_flags_statements_speculation_quotes_and_length(headline):
    assert check_headline(headline)


@pytest.mark.parametrize("headline", [
    "Francia impone toques de queda parciales en París y otras ciudades",
    "Trump propone una cumbre con Putin y Xi en noviembre",
    "Milei prevé un superávit comercial de US$30.000 millones",
    "El Gobierno declara el estado de emergencia en tres regiones",  # 'declara' is a legal act
    "Casación confirma el archivo de la causa contra Macri",
])
def test_check_headline_accepts_factual_headlines(headline):
    assert check_headline(headline) == []


def test_check_lines_enforces_count_length_and_no_talk_about_sources():
    assert check_lines(["Una.", "Dos."]) == []
    assert any("exactamente 2" in p for p in check_lines(["Una.", "Dos.", "Tres."]))
    assert any("palabras" in p for p in check_lines(["una dos tres cuatro cinco seis siete ocho nueve diez once doce trece", "Dos."]))
    assert check_lines(["Según los medios, algo.", "El Gobierno no ha respondido."])


def test_check_numbers_ignores_formatting_and_accepts_mil():
    srcs = [{"name": "A", "title": "Cerca de mil agentes heridos", "snippet": "Toque de queda de 23:00 a 06:00; 2.400 detenidos"}]
    assert check_numbers("Francia", ["1.000 agentes heridos", "Toque de 23.00 a 6.00; 2400 detenidos"], srcs) == []


def test_check_numbers_flags_invented_figures():
    srcs = [{"name": "A", "title": "Dos mil detenidos", "snippet": "balance de 2.400 detenidos"}]
    problems = check_numbers("Francia", ["3.100 detenidos"], srcs)
    assert problems and "3.100" in problems[0]


# --- flow -----------------------------------------------------------------------------------

def test_good_summary_costs_one_generation_and_one_verification():
    client = FakeClient()
    headline, text = summarize_story(client, story())
    assert (client.gen_calls, client.verify_calls) == (1, 1)
    assert headline == GOOD["headline"] and text.split("\n") == GOOD["lines"]


def test_format_violation_retries_without_spending_a_verification_call():
    bad = {"headline": "Trump se declara dispuesto", "lines": ["Una.", "Dos."]}
    client = FakeClient(drafts=[bad, GOOD])
    summarize_story(client, story())
    assert client.gen_calls == 2 and client.verify_calls == 1  # only the valid draft was verified
    assert "se declara" in client.prompts[1]


def test_verifier_problems_are_fed_back_and_the_corrected_draft_is_kept():
    inverted = {"headline": "Cargas policiales frenan la salida de Ayuso", "lines": ["Una.", "Dos."]}
    client = FakeClient(drafts=[inverted, GOOD],
                        verdicts=[["la Policía carga para facilitar la salida, no para frenarla"], []])
    headline, _ = summarize_story(client, story())
    assert headline == GOOD["headline"]
    assert "facilitar la salida" in client.prompts[1]


def test_raises_when_no_attempt_passes():
    client = FakeClient(verdicts=[["afirmación sin respaldo"]])
    with pytest.raises(UnverifiedSummary):
        summarize_story(client, story())
    assert client.gen_calls == 3


# --- storage --------------------------------------------------------------------------------

def test_summarize_stores_lines_and_caches(conn):
    client = FakeClient()
    assert summarize_stories(conn, [story()], client) == 1
    row = conn.execute("SELECT * FROM summary").fetchone()
    assert (row["headline"], row["model"]) == (GOOD["headline"], SUMMARY_TAG)
    assert row["text"].split("\n") == GOOD["lines"]
    assert summarize_stories(conn, [story()], client) == 0  # unchanged cluster: nothing to do
    assert client.gen_calls == 1


def test_old_format_summaries_are_rebuilt(conn):
    conn.execute("INSERT INTO summary (cluster_id, model, n_members, headline, text, created_at) "
                 "VALUES (1, 'claude-haiku-5-5', 3, 'old', 'old text', 'x')")
    assert summarize_stories(conn, [story()], FakeClient()) == 1
    assert conn.execute("SELECT model FROM summary").fetchone()[0] == SUMMARY_TAG


def test_unverified_story_keeps_previous_summary_and_does_not_stop_the_run(conn):
    summarize_stories(conn, [story(n=3)], FakeClient())
    bad = FakeClient(verdicts=[["sin respaldo"]])
    summarize_stories(conn, [story(n=9)], bad)  # grew enough to refresh, but nothing passes
    assert conn.execute("SELECT headline FROM summary").fetchone()["headline"] == GOOD["headline"]


def test_api_failures_stop_after_consecutive_errors(conn):
    bad = FakeClient(fail=True)
    summarize_stories(conn, [story(cid=i) for i in range(1, 8)], bad)
    assert conn.execute("SELECT COUNT(*) FROM summary").fetchone()[0] == 0
