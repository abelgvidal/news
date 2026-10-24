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


def _c(cid, vec, size=1, country="es"):
    c = cluster(cid, vec)
    c.size, c.country = size, country
    return c


def test_plan_merges_fuses_near_identical_and_keeps_the_larger():
    from agregador.cluster import plan_merges

    big, small = _c(1, unit(1, 0.02), size=8), _c(2, unit(1, 0.0), size=3)
    other = _c(3, unit(0, 1), size=5)
    merges = plan_merges([big, small, other], threshold=0.965)
    assert [(k, d) for k, d, _ in merges] == [(1, 2)]
    assert big.size == 11


def test_plan_merges_never_crosses_countries_or_fuses_distinct_stories():
    from agregador.cluster import plan_merges

    same_vec_other_country = [_c(1, unit(1, 0), country="es"), _c(2, unit(1, 0), country="ar")]
    assert plan_merges(same_vec_other_country, 0.965) == []
    distinct = [_c(1, unit(1, 0.4)), _c(2, unit(1, 0.0))]  # sim ~0.93 < 0.965
    assert plan_merges(distinct, 0.965) == []


def test_merge_clusters_moves_members_and_deletes_the_dropped_cluster(tmp_path):
    from agregador import db
    from agregador.cluster import merge_clusters

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    conn.execute("INSERT INTO feed (id, name, url) VALUES (1, 'A', 'a')")
    for cid, aid, n in [(1, 1, 2), (2, 2, 1)]:
        conn.execute(
            "INSERT INTO cluster (id, model, vector_sum, size, feed_ids, created_at, last_published_at) "
            "VALUES (?, 'm', ?, ?, '[1]', 'x', '2026-10-19T10:00:00+00:00')",
            (cid, (unit(1, 0) * n).tobytes(), n))
        conn.execute(
            "INSERT INTO article (id, feed_id, url, title, published_at, fetched_at) "
            "VALUES (?, 1, ?, 't', '2026-10-19T10:00:00+00:00', 'x')", (aid, f"u{aid}"))
        conn.execute("INSERT INTO cluster_member (article_id, cluster_id) VALUES (?, ?)", (aid, cid))
    assert merge_clusters(conn, "m", Params(merge_threshold=0.965)) == 1
    assert [r[0] for r in conn.execute("SELECT id FROM cluster")] == [1]
    rows = conn.execute("SELECT cluster_id, note FROM cluster_member ORDER BY article_id").fetchall()
    assert [r["cluster_id"] for r in rows] == [1, 1]
    assert "merged from cluster 2" in rows[1]["note"]
    assert conn.execute("SELECT size FROM cluster").fetchone()[0] == 3


def test_two_big_clusters_need_the_stricter_threshold_to_fuse():
    from agregador.cluster import plan_merges

    a, b = _c(1, unit(1, 0.0), size=12), _c(2, unit(1, 0.2), size=10)   # sim ~0.98
    assert plan_merges([a, b], 0.965, small_size=8, big_threshold=0.985) == []
    c, d = _c(1, unit(1, 0.0), size=12), _c(2, unit(1, 0.05), size=10)  # sim ~0.9988
    assert [(k, x) for k, x, _ in plan_merges([c, d], 0.965, small_size=8, big_threshold=0.985)] == [(1, 2)]


def test_a_small_cluster_is_absorbed_at_the_normal_threshold():
    from agregador.cluster import plan_merges

    big, small = _c(1, unit(1, 0.0), size=40), _c(2, unit(1, 0.2), size=3)  # sim ~0.98
    assert [(k, x) for k, x, _ in plan_merges([big, small], 0.965, small_size=8, big_threshold=0.985)] == [(1, 2)]


def test_fusions_cannot_chain_into_one_giant_cluster():
    from agregador.cluster import plan_merges

    # A-B and B-C are each close, A-C is not: absorbing B into A must not then drag C in
    a, b, c = _c(1, unit(1, 0.0), size=20), _c(2, unit(1, 0.25), size=3), _c(3, unit(1, 0.5), size=15)
    merges = plan_merges([a, b, c], 0.965, small_size=8, big_threshold=0.985)
    assert len(merges) == 1                      # b goes to one of them, and that is all
    assert {merges[0][0], merges[0][1]} != {1, 3}  # the two big clusters never fuse


def test_an_article_dated_in_the_future_does_not_switch_off_the_merging(tmp_path):
    """Regression: one article dated days ahead made `latest` jump, so no real cluster counted as active."""
    from datetime import timedelta

    from agregador import db
    from agregador.cluster import merge_clusters

    now = datetime(2026, 10, 24, 5, 0, tzinfo=UTC)
    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    conn.execute("INSERT INTO feed (id, name, url) VALUES (1, 'A', 'a')")
    last = {1: now - timedelta(hours=10), 2: now - timedelta(hours=9), 3: now + timedelta(days=3)}  # 3: the future one
    for cid, vec in [(1, unit(1, 0)), (2, unit(1, 0)), (3, unit(0, 1))]:
        conn.execute(
            "INSERT INTO cluster (id, model, vector_sum, size, feed_ids, created_at, last_published_at) "
            "VALUES (?, 'm', ?, 1, '[1]', 'x', ?)", (cid, vec.tobytes(), last[cid].isoformat()))
        conn.execute(
            "INSERT INTO article (id, feed_id, url, title, published_at, fetched_at) "
            "VALUES (?, 1, ?, 't', ?, 'x')", (cid, f"u{cid}", last[cid].isoformat()))
        conn.execute("INSERT INTO cluster_member (article_id, cluster_id) VALUES (?, ?)", (cid, cid))
    assert merge_clusters(conn, "m", Params(merge_threshold=0.965), now=now) == 1
