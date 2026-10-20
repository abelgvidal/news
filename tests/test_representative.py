from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from agregador.representative import centrality, headline_penalties, pick_representative

T0 = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)


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
