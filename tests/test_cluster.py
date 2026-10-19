import numpy as np

from agregador.cluster import Cluster, Params, choose, extract_entities
from datetime import datetime, UTC


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def cluster(cid, vec, feeds=(1,), entities=()):
    return Cluster(cid, vec.copy(), 1, set(entities), set(feeds), datetime.now(UTC))


P = Params(threshold=0.9, same_feed_threshold=0.97, entity_free_threshold=0.95)


def test_joins_best_cluster_above_threshold():
    c1, c2 = cluster(1, unit(1, 0)), cluster(2, unit(0, 1))
    r = choose(unit(1, 0.05), {"puigdemont"}, 9, [c1, c2], P)
    assert r.cluster_id == 1


def test_starts_new_cluster_below_threshold():
    r = choose(unit(1, 1), set(), 9, [cluster(1, unit(1, 0))], P)
    assert r.cluster_id is None


def test_same_outlet_needs_higher_similarity():
    c = cluster(1, unit(1, 0), feeds=(5,))
    assert choose(unit(1, 0.3), set(), 5, [c], P).cluster_id is None  # sim ~0.96 < 0.97
    assert choose(unit(1, 0.3), set(), 6, [c], P).cluster_id == 1


def test_entity_guard_blocks_unrelated_borderline_match():
    c = cluster(1, unit(1, 0), entities={"francia", "paris"})
    assert choose(unit(1, 0.4), {"melissa", "parke"}, 9, [c], P).cluster_id is None
    assert choose(unit(1, 0.4), {"paris"}, 9, [c], P).cluster_id == 1


def test_extract_entities_skips_title_start_and_stopwords():
    ents = extract_entities("Feijóo firma su Contrato con España", "El PP presenta 12 medidas en Madrid")
    assert {"contrato", "madrid", "12"} <= ents
    assert "feijóo" not in ents and "españa" not in ents


def test_clusters_never_mix_countries(tmp_path):
    import sqlite3
    from agregador import db
    from agregador.cluster import cluster_pending

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    conn.execute("INSERT INTO feed (id, name, url, country) VALUES (1, 'A', 'a', 'es')")
    conn.execute("INSERT INTO feed (id, name, url, country) VALUES (2, 'B', 'b', 'ar')")
    conn.execute("INSERT INTO feed (id, name, url, country) VALUES (3, 'C', 'c', 'es')")
    vec = unit(1, 0).tobytes()  # identical embedding everywhere: only the country can keep them apart
    for aid, feed in [(1, 1), (2, 2), (3, 3)]:
        conn.execute(
            "INSERT INTO article (id, feed_id, url, title, summary, published_at, fetched_at) "
            "VALUES (?, ?, ?, 'Titular', '', '2026-10-19T10:00:00+00:00', 'x')", (aid, feed, f"u{aid}"))
        conn.execute("INSERT INTO embedding (article_id, model, vector) VALUES (?, 'm', ?)", (aid, vec))
    cluster_pending(conn, "m", Params(threshold=0.9))
    rows = conn.execute(
        "SELECT c.country, COUNT(*) n FROM cluster c JOIN cluster_member m ON m.cluster_id = c.id "
        "GROUP BY c.id ORDER BY c.country").fetchall()
    assert [(r["country"], r["n"]) for r in rows] == [("ar", 1), ("es", 2)]
