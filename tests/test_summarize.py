import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agregador import db
from agregador.render import Story
from agregador.summarize import (
    ESCALATION_MODEL, NotAStory, SUMMARY_MODEL, SUMMARY_TAG, UnverifiedSummary, Usage, VERIFY_MODEL,
    VERIFY_SYSTEM, build_prompt, check_coverage, check_headline, check_lines, check_numbers, needs_refresh,
    parse_response, run_stats, select_sources, summarize_stories, summarize_story, valid_outlets,
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
        self.prompts, self.models = [], []
        self.messages = self

    def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        self.models.append(kwargs["model"])
        if kwargs["system"] == VERIFY_SYSTEM:
            payload = {"problemas": self.verdicts[min(self.verify_calls, len(self.verdicts) - 1)]}
            self.verify_calls += 1
        else:
            prompt = kwargs["messages"][0]["content"]
            payload = dict(self.drafts[min(self.gen_calls, len(self.drafts) - 1)])
            payload.setdefault("outlets", re.findall(r"^- \[([^\]]+)\]", prompt, re.M))
            self.gen_calls += 1
            self.prompts.append(prompt)
        block = SimpleNamespace(type="text", text=json.dumps(payload))
        return SimpleNamespace(stop_reason="end_turn", content=[block],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


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
    assert parse_response('{"headline": " A ", "lines": [" B ", ""], "outlets": [" X "]}') == ("A", ["B"], ["X"])
    assert parse_response('{"headline": "A", "lines": ["B"]}')[2] == []
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
    headline, text, covering = summarize_story(client, story())
    assert (client.gen_calls, client.verify_calls) == (1, 1)
    assert headline == GOOD["headline"] and text.split("\n") == GOOD["lines"]
    assert covering == ["M0", "M1", "M2"]


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
    headline, _, _ = summarize_story(client, story())
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


# --- models, escalation, usage ---------------------------------------------------------------

def test_cheap_model_writes_the_stronger_model_verifies():
    client = FakeClient()
    summarize_story(client, story())
    assert client.models == [SUMMARY_MODEL, VERIFY_MODEL]


def test_the_last_attempt_is_written_by_the_stronger_model():
    client = FakeClient(verdicts=[["afirmación sin respaldo"]])
    with pytest.raises(UnverifiedSummary):
        summarize_story(client, story())
    assert client.models == [SUMMARY_MODEL, VERIFY_MODEL, SUMMARY_MODEL, VERIFY_MODEL,
                             ESCALATION_MODEL, VERIFY_MODEL]


def test_usage_counts_calls_tokens_and_cost():
    usage, client = Usage(), FakeClient(verdicts=[["x"], []])
    summarize_story(client, story(), usage)
    assert (usage.gen_calls, usage.verify_calls, usage.escalations) == (2, 2, 0)
    assert usage.tokens[SUMMARY_MODEL] == [2000, 200] and usage.tokens[VERIFY_MODEL] == [2000, 200]
    # haiku: 2000*0.10/1e6 + 200*0.50/1e6 ; sonnet: 2000*2/1e6 + 200*10/1e6
    assert usage.cost_usd == pytest.approx(0.0003 + 0.006)


def test_runs_are_recorded_and_aggregated(conn):
    summarize_stories(conn, [story()], FakeClient())
    summarize_stories(conn, [story(n=9)], FakeClient(verdicts=[["sin respaldo"]]))  # nothing passes
    row = conn.execute("SELECT * FROM summary_run ORDER BY id").fetchall()
    assert [(r["attempted"], r["stored"], r["unverified"]) for r in row] == [(1, 1, 0), (1, 0, 1)]
    st = run_stats(conn, days=7)
    assert (st["attempted"], st["stored"], st["unverified"]) == (2, 1, 1)
    assert st["fallback_rate"] == 0.5 and st["gen"] == 4 and st["esc"] == 1 and st["cost"] > 0


def test_nothing_attempted_records_no_run(conn):
    summarize_stories(conn, [story()], FakeClient())
    summarize_stories(conn, [story()], FakeClient())  # cached
    assert conn.execute("SELECT COUNT(*) FROM summary_run").fetchone()[0] == 1


def test_last_attempt_over_length_is_kept_when_verified():
    long_lines = {"headline": "Francia impone toques de queda en París",
                  "lines": ["una dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce", "Dos."]}
    client = FakeClient(drafts=[long_lines])  # every draft runs long; the verifier finds nothing wrong
    headline, text, _ = summarize_story(client, story())
    assert headline == long_lines["headline"] and "catorce" in text
    assert client.gen_calls == 3 and client.verify_calls == 1  # only the last, tolerated draft is verified


def test_over_length_never_excuses_a_real_problem():
    quoted = {"headline": 'Feijóo: "no voy a fallar"', "lines": ["Una.", "Dos."]}
    with pytest.raises(UnverifiedSummary):
        summarize_story(FakeClient(drafts=[quoted]), story())


def test_last_attempt_over_length_still_fails_if_verification_fails():
    long_lines = {"headline": "Titular válido aquí",
                  "lines": ["una dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce", "Dos."]}
    with pytest.raises(UnverifiedSummary):
        summarize_story(FakeClient(drafts=[long_lines], verdicts=[["sin respaldo"]]), story())


def test_digits_glued_to_letters_are_names_not_figures():
    srcs = [{"name": "A", "title": "Cuatro muertos", "snippet": "balance de 4 fallecidos"}]
    assert check_numbers("Dana", ["Cinco muertos según 20minutos", "Elecciones el 29N"], srcs) == []
    assert check_numbers("Dana", ["Cinco muertos según 20 medios"], srcs)  # a bare number is still checked


def test_a_failed_story_is_not_retried_until_the_cooldown_passes_or_the_cluster_grows(conn):
    nothing_passes = FakeClient(verdicts=[["sin respaldo"]])
    assert summarize_stories(conn, [story(n=3)], nothing_passes) == 1
    assert conn.execute("SELECT COUNT(*) FROM summary_failure").fetchone()[0] == 1
    again = FakeClient()
    assert summarize_stories(conn, [story(n=3)], again) == 0 and again.gen_calls == 0   # cooling down
    assert summarize_stories(conn, [story(n=9)], again) == 1                            # it grew: try again
    assert conn.execute("SELECT COUNT(*) FROM summary_failure").fetchone()[0] == 0     # success clears it
    assert conn.execute("SELECT COUNT(*) FROM summary").fetchone()[0] == 1


def test_cooldown_expires(conn):
    conn.execute("INSERT INTO summary_failure (cluster_id, n_members, failed_at, reason) "
                 "VALUES (1, 3, '2020-01-01T00:00:00+00:00', 'x')")
    assert summarize_stories(conn, [story(n=3)], FakeClient()) == 1


# --- main event and covering outlets ---------------------------------------------------------

def test_valid_outlets_keeps_only_real_outlets_with_canonical_spelling():
    assert valid_outlets(["el país", "Inventado", "El País", "ABC"], {"El País", "ABC", "COPE"}) == ["El País", "ABC"]


def test_check_coverage_needs_three_outlets_and_half_of_those_seen():
    assert check_coverage(["A", "B", "C"], 5) == []
    assert check_coverage(["A", "B"], 2)                      # fewer than three
    assert check_coverage(["A", "B", "C"], 8)                 # three of eight is a bag of topics


def test_too_little_coverage_is_fed_back_and_a_better_draft_is_accepted():
    narrow = {**GOOD, "outlets": ["M0"]}
    client = FakeClient(drafts=[narrow, GOOD])
    _, _, covering = summarize_story(client, story(n=4))
    assert client.gen_calls == 2 and client.verify_calls == 1  # the narrow draft was never verified
    assert "solo 1 de los 4 medios" in client.prompts[1] and covering == ["M0", "M1", "M2", "M3"]


def test_a_bag_of_topics_is_not_a_story_and_costs_no_verification():
    client = FakeClient(drafts=[{**GOOD, "outlets": ["M0"]}])
    with pytest.raises(NotAStory):
        summarize_story(client, story(n=6))
    assert client.gen_calls == 3 and client.verify_calls == 0


def test_not_a_story_is_stored_as_skipped_and_not_retried(conn):
    client = FakeClient(drafts=[{**GOOD, "outlets": ["M0"]}])
    assert summarize_stories(conn, [story(n=6)], client) == 1
    row = conn.execute("SELECT status, headline FROM summary").fetchone()
    assert (row["status"], row["headline"]) == ("skip", "")
    assert summarize_stories(conn, [story(n=6)], client) == 0  # remembered, no new calls
    assert conn.execute("SELECT skipped FROM summary_run").fetchone()[0] == 1


def test_covering_outlets_are_stored(conn):
    summarize_stories(conn, [story(n=4)], FakeClient())
    assert json.loads(conn.execute("SELECT outlets FROM summary").fetchone()[0]) == ["M0", "M1", "M2", "M3"]


def test_the_fill_target_stops_early_and_skipped_stories_do_not_count(conn):
    for cid in range(2, 6):
        conn.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at) "
                     "VALUES (?, 'm', x'00', 3, 'x', 'x')", (cid,))
    skip_client = FakeClient(drafts=[{**GOOD, "outlets": ["M0"]}])
    summarize_stories(conn, [story(cid=1, n=6)], skip_client)           # cluster 1 ends up skipped
    client = FakeClient()
    attempted = summarize_stories(conn, [story(cid=c, n=3) for c in range(1, 6)], client, target=2)
    assert attempted == 2                                               # 2 and 3 fill the target of 2
    assert conn.execute("SELECT COUNT(*) FROM summary WHERE status='ok'").fetchone()[0] == 2
