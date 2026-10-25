from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from agregador.representative import (
    centrality, choose_story, core_members, headline_penalties, headline_to_show, title_overlap, title_words,
)

T0 = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)


def pick_representative(titles, vecs, published):
    return choose_story(titles, vecs, published)[0]


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("title", [
    'Feijóo: "No voy a fallar"',
    "Feijóo: «No voy a fallar»",
    "¿Quién gana las elecciones?",
    "Última hora de las elecciones, en directo",
    "Sánchez, EN VIVO: todo sobre el mitin",
    "Dana Adam | Última hora sobre la lluvia",
    "SÁNCHEZ DISUELVE LAS CORTES Y CONVOCA ELECCIONES",
    "Opinión: lo que se juega el país",
    "x" * 120,
])
def test_poor_front_page_headlines_are_penalized(title):
    assert headline_penalties(title) >= 1


@pytest.mark.parametrize("title", [
    "Sánchez disuelve las Cortes y convoca elecciones el 29 de noviembre",
    "Feijóo firma su 'Contrato con España' con doce compromisos",
    "La prima de riesgo francesa sube a máximos de 2012",
])
def test_plain_news_headlines_are_not_penalized(title):
    assert headline_penalties(title) == 0


def test_centrality_is_the_mean_similarity_to_the_others():
    v = np.stack([unit(1, 0), unit(1, 0.1), unit(0, 1)])
    c = centrality(v)
    assert c[1] > c[2] and c[0] > c[2]          # the outlier is the least central
    assert centrality(np.stack([unit(1, 0)])).tolist() == [1.0]


def test_the_most_central_article_represents_the_cluster_not_the_outlier():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.05), unit(1, 0.1), unit(0.5, 1)])   # last one drifts off
    titles = ["A", "B", "C", "D"]
    best = pick_representative(titles, vecs, [T0 + timedelta(minutes=i) for i in range(4)])
    assert best in (1, 2) and best != 3


def test_a_live_blog_loses_to_a_plain_headline_of_a_similar_article():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.01)])
    titles = ["Última hora de la dana, en directo", "La dana Adam deja cuatro muertos"]
    assert pick_representative(titles, vecs, [T0, T0 + timedelta(hours=1)]) == 1


def test_a_clearly_more_central_article_can_beat_a_small_penalty_only_if_it_is_much_closer():
    # centrality gaps are tiny, so a penalty (0.03) outweighs them: the quoted headline loses here
    vecs = np.stack([unit(1, 0.0), unit(1, 0.02), unit(1, 0.04)])
    titles = ['"Cita" textual', "Titular plano uno", "Titular plano dos"]
    assert pick_representative(titles, vecs, [T0] * 3) in (1, 2)


def test_ties_go_to_the_article_published_first():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.0)])
    assert pick_representative(["Primero", "Segundo"], vecs, [T0, T0 + timedelta(hours=2)]) == 0
    assert pick_representative(["Primero", "Segundo"], vecs, [T0 + timedelta(hours=2), T0]) == 1


def test_a_penalty_never_pushes_the_choice_to_a_clearly_less_central_article():
    # the plain headline drifts off the story: penalties must not rescue it from being passed over
    vecs = np.stack([unit(1, 0.0), unit(1, 0.02), unit(1, 0.05), unit(0.6, 1)])
    titles = ['"Cita" textual', "Plano uno", "Plano dos", "Plano lejano"]
    best = pick_representative(titles, vecs, [T0] * 4)
    assert best != 3 and best in (1, 2)


def test_title_overlap_counts_shared_content_words_only():
    assert title_overlap("Feijóo firma su contrato con España", "El contrato de Feijóo con España") > 0.4
    assert title_overlap("La dana deja cuatro muertos", "Meloni envía 500 militares a Milán") == 0.0


def test_an_english_possessive_does_not_hide_a_shared_word():
    """Regression: "Brazil’s" became "brazils", so it shared nothing with "Brazil" and a borderline outlet fell out of the core."""
    assert title_words("Brazil’s Lula") == title_words("Brazil Lula") == {"brazil", "lula"}
    assert title_words("The company's board") == {"company", "board"}
    assert title_overlap("Brazil’s Lula debates", "Brazil candidates debate") > 0.0


def test_core_keeps_the_story_and_drops_neighbouring_topics():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.05), unit(1, 0.9)])       # the last one is a different topic
    titles = ["Meloni envía 500 militares a Milán", "Meloni despliega al Ejército en Milán", "Qué dice Francia de Europa"]
    assert core_members(titles, vecs, 0) == [0, 1]


def test_shared_words_rescue_a_borderline_article_that_embeddings_alone_would_drop():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.54)])                      # similarity ~0.88: under the 0.90 bar on its own
    assert 0.87 < float(vecs[0] @ vecs[1]) < 0.89
    rep = "Abascal ofrece al PP un pacto en el Senado"
    assert core_members([rep, "Algo muy distinto aquí"], vecs, 0) == [0]
    assert core_members([rep, "Pacto en el Senado: Abascal ofrece al PP candidatos"], vecs, 0) == [0, 1]


def test_a_live_blog_that_is_the_most_central_still_loses_to_a_plain_headline_of_the_same_story():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.1), unit(1, -0.1)])      # the live blog sits right in the middle
    titles = ["Milei, EN VIVO: todo sobre el discurso", "Milei habla en Neuquén ante empresarios", "Milei en la Expo de Neuquén"]
    rep, core = choose_story(titles, vecs, [T0 + timedelta(hours=2), T0, T0 + timedelta(hours=1)])
    assert rep == 1 and core == [0, 1, 2]


def test_the_representative_always_belongs_to_the_core_and_drifted_articles_are_out():
    vecs = np.stack([unit(1, 0.0), unit(1, 0.05), unit(1, 0.9)])
    titles = ["Meloni envía 500 militares a Milán", "Meloni despliega al Ejército en Milán", "Qué dice Francia de Europa"]
    rep, core = choose_story(titles, vecs, [T0] * 3)
    assert rep in core and 2 not in core


def test_a_penalty_never_picks_a_tangential_article_over_the_central_ones():
    # every central headline carries a speech quote; the only plain one is clearly off to the side
    vecs = np.stack([unit(1, 0.0), unit(1, 0.04), unit(1, -0.04), unit(1, 0.45)])
    titles = ['Milei: "primera cita"', 'Milei: "segunda cita"', 'Milei: "tercera cita"', "Milei viajó a España"]
    rep, core = choose_story(titles, vecs, [T0] * 4)
    assert rep != 3        # the plain headline is not central enough to be a candidate


# --- the headline shown: the representative's, unless it is an outlier in length for its own story ---

EAGLES_NEST = [   # the nine headlines of the story that opened the 2026-10-25 International edition
    "Fire engulfs Hitler’s former Eagle’s Nest retreat in Germany",
    "Suspect detained after fire wrecks Hitler’s mountain retreat",
    "Watch as fire engulfs Hitler's mountaintop Eagle's Nest retreat",
    "Man arrested after fire breaks out at Hitler’s former Alpine retreat",
    "‘Eagle’s Nest,’ Nazi-Era Guesthouse in German Alps, Is Gutted by Fire",
    "German police detain suspect after fire engulfs Hitler’s Eagle’s Nest",
    "German man detained after fire wrecks Hitler’s ‘Eagle’s Nest’ mountain retreat",
    "Suspect arrested as fire engulfs Nazi guest house 'Eagle's Nest', German police say",
    "Suspect arrested on suspicion of arson after fire engulfs Hitler’s Eagle’s Nest mountain retreat in Germany",
]


def test_a_headline_far_longer_than_its_story_s_usual_one_is_swapped_for_a_shorter_one_of_the_same_story():
    core = list(range(len(EAGLES_NEST)))
    shown = headline_to_show(EAGLES_NEST, core, rep=8)
    assert EAGLES_NEST[shown] == "German police detain suspect after fire engulfs Hitler’s Eagle’s Nest"


def test_a_headline_that_is_only_somewhat_longer_than_usual_is_left_alone():
    titles = ["Powerful earthquake hits Tibet killing 16 people", "Earthquake kills at least 16 people in Tibet",
              "Strong quake in Tibet leaves 16 people dead", "Tibet earthquake kills at least 16, injures dozens",
              "Tibet earthquake kills at least 16 people, injures dozens"]
    longest = max(range(5), key=lambda i: len(titles[i]))
    assert len(titles[longest]) < 1.35 * sorted(map(len, titles))[2]
    assert headline_to_show(titles, list(range(5)), rep=longest) == longest


def test_the_swap_never_drops_a_figure_the_long_headline_has():
    titles = ["Earthquake in Tibet kills at least 16 people and destroys hundreds of homes in remote villages",
              "Tibet earthquake kills 16", "Earthquake kills 16 in Tibet", "Strong earthquake strikes Tibet", "Tibet earthquake"]
    shown = headline_to_show(titles, list(range(5)), rep=0)
    assert shown != 0 and "16" in titles[shown] and "kills" in titles[shown]


def test_a_shorter_headline_with_a_question_mark_is_not_a_replacement():
    titles = ["Fire engulfs the old mountain retreat after a suspect was arrested on suspicion of arson in the Alps",
              "Fire engulfs the old mountain retreat?", "Fire engulfs the old mountain retreat", "Mountain retreat fire engulfs",
              "Old mountain retreat engulfs fire"]
    shown = headline_to_show(titles, list(range(5)), rep=0)
    assert "?" not in titles[shown] and shown != 0


def test_a_story_of_fewer_than_four_headlines_keeps_its_representative():
    titles = ["Fire engulfs the retreat", "Fire engulfs the old mountain retreat after a suspect was arrested on suspicion of arson",
              "Fire at the retreat"]
    assert headline_to_show(titles, [0, 1, 2], rep=1) == 1
