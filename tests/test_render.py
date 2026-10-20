from datetime import UTC, datetime, timedelta

import numpy as np

from agregador import db
from agregador.config import Country
from agregador.render import Story, build_stories, render_page, score, top_stories

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def story(cid, outlets, hours_ago, headline="Titular <b>"):
    t = NOW - timedelta(hours=hours_ago)
    sources = [{"name": f"M{i}", "title": headline if i == 0 else f"otro {i}", "url": f"https://e.com/{cid}/{i}",
                "published_at": t, "similarity": 1.0 - i * 0.01, "image_url": None} for i in range(outlets)]
    return Story(cid, headline, sources[0]["url"], "M0", None, t, score(outlets, t, NOW), sources)


def test_score_prefers_more_outlets_and_fresher():
    assert score(5, NOW, NOW) > score(3, NOW, NOW)
    assert score(3, NOW, NOW) > score(3, NOW - timedelta(hours=24), NOW)


def test_the_front_page_has_only_the_ten_most_covered_stories_and_none_with_a_single_outlet():
    stories = [story(i, 2 + i % 5, i) for i in range(1, 16)] + [story(99, 1, 0)]
    assert len(top_stories(stories)) == 10 and all(s.cluster_id != 99 for s in top_stories(stories))
    html = render_page(stories, NOW)
    assert html.count('<article class="story') == 10


def test_the_headline_is_shown_as_published_linked_to_its_article_with_its_outlet():
    html = render_page([story(1, 3, 1, headline='Feijóo: "No voy a fallar" & más')], NOW)
    assert 'href="https://e.com/1/0"' in html and 'rel="noopener noreferrer nofollow"' in html
    assert "Feijóo: &#34;No voy a fallar&#34; &amp; más" in html        # untouched, only escaped
    assert '<p class="source">M0</p>' in html


def test_html_in_a_headline_is_escaped():
    html = render_page([story(1, 3, 1)], NOW)
    assert "&lt;b&gt;" in html and "Titular <b>" not in html


def test_other_outlets_exclude_the_representatives_and_fold_after_four():
    s = story(1, 7, 1)                       # M0 is the representative; M1..M6 are the others
    assert [o["name"] for o in s.other_outlets] == ["M1", "M2", "M3", "M4", "M5", "M6"]
    html = render_page([s], NOW)
    assert html.count("<details") == 1 and "y 2 más" in html
    shown, folded = html[:html.index("<details")], html[html.index("<details"):]
    assert all(f'href="https://e.com/1/{i}"' in shown for i in (1, 2, 3, 4))
    assert all(f'href="https://e.com/1/{i}"' in folded for i in (5, 6))
    assert 'href="https://e.com/1/0"' in shown                          # the headline itself


def test_few_outlets_need_no_toggle():
    assert "<details" not in render_page([story(1, 3, 1)], NOW)


def test_only_http_urls_become_links_and_unsafe_ones_are_dropped():
    s = story(1, 3, 1)
    s.sources[1]["url"] = "javascript:alert(1)"
    s.sources[2]["url"] = "data:text/html,<script>"
    s.url = None                              # the representative's own URL was unsafe
    html = render_page([s], NOW)
    assert "javascript:" not in html and "data:text" not in html
    assert "M1" in html and "M2" in html      # still listed, just not linked
    assert '<h2><a' not in html               # the headline is plain text then


def test_tabs_link_relative_to_each_page():
    es, ar = Country("es", "España", "Europe/Madrid"), Country("ar", "Argentina", "America/Argentina/Buenos_Aires")
    home = render_page([story(1, 3, 1)], NOW, es, [es, ar], root="")
    sub = render_page([story(1, 3, 1)], NOW, ar, [es, ar], root="../")
    assert 'href="index.html"' in home and 'href="ar/index.html"' in home
    assert 'href="../index.html"' in sub and 'href="../ar/index.html"' in sub
    assert 'aria-current="page">Argentina' in sub


def test_single_country_has_no_tabs():
    assert "<nav" not in render_page([story(1, 3, 1)], NOW)


# --- building stories from the database ------------------------------------------------------

def _db_with_cluster(tmp_path, articles):
    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    conn.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at, country) "
                 "VALUES (1, 'm', x'00', ?, 'x', ?, 'es')", (len(articles), NOW.isoformat()))
    for i, (name, title, vec, image, minutes) in enumerate(articles, 1):
        conn.execute("INSERT OR IGNORE INTO feed (name, url, country) VALUES (?, ?, 'es')", (name, f"f-{name}"))
        fid = conn.execute("SELECT id FROM feed WHERE name = ?", (name,)).fetchone()[0]
        conn.execute("INSERT INTO article (id, feed_id, url, title, image_url, published_at, fetched_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, 'x')",
                     (i, fid, f"https://{name}.example/{i}", title, image,
                      (NOW - timedelta(minutes=minutes)).isoformat()))
        conn.execute("INSERT INTO embedding (article_id, model, vector) VALUES (?, 'm', ?)",
                     (i, np.array(vec, dtype=np.float32).tobytes()))
        conn.execute("INSERT INTO cluster_member (article_id, cluster_id) VALUES (?, 1)", (i,))
    return conn


def _u(*xs):
    v = np.array(xs, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_the_story_is_the_most_central_article_with_its_own_headline_and_outlet(tmp_path):
    conn = _db_with_cluster(tmp_path, [
        ("A", "Titular central", _u(1, 0.02), None, 30),
        ("B", "Titular parecido", _u(1, 0.0), None, 20),
        ("C", "Titular cercano", _u(1, 0.08), None, 10),
    ])
    (s,) = build_stories(conn, "m", NOW, "es")
    assert (s.headline, s.source, s.outlets) == ("Titular central", "A", 3)
    assert s.sources[0]["name"] == "A"                                    # representative first
    assert s.url == "https://A.example/1"


def test_an_article_from_a_neighbouring_topic_is_left_out_of_the_story_and_of_the_count(tmp_path):
    conn = _db_with_cluster(tmp_path, [
        ("A", "Titular central", _u(1, 0.02), None, 30),
        ("B", "Titular parecido", _u(1, 0.0), None, 20),
        ("C", "Otro asunto distinto", _u(0.4, 1), None, 10),             # drifted into the cluster
    ])
    (s,) = build_stories(conn, "m", NOW, "es")
    assert s.outlets == 2 and [x["name"] for x in s.sources] == ["A", "B"]
    assert "C" not in [o["name"] for o in s.other_outlets]


def test_a_quoted_headline_is_not_chosen_when_a_plain_one_is_as_good(tmp_path):
    conn = _db_with_cluster(tmp_path, [
        ("A", 'Feijóo: "No voy a fallar"', _u(1, 0.0), None, 30),
        ("B", "Feijóo firma su contrato con doce compromisos", _u(1, 0.01), None, 20),
    ])
    assert build_stories(conn, "m", NOW, "es")[0].headline.startswith("Feijóo firma")


def test_the_image_is_the_representatives_or_the_closest_article_that_has_one(tmp_path):
    conn = _db_with_cluster(tmp_path, [
        ("A", "Central sin imagen", _u(1, 0.02), None, 30),
        ("B", "Cercano con imagen", _u(1, 0.0), "https://img.example/b.jpg", 20),
        ("C", "Cercano con imagen también", _u(1, 0.1), "https://img.example/c.jpg", 10),
    ])
    assert build_stories(conn, "m", NOW, "es")[0].image_url == "https://img.example/b.jpg"
    conn.execute("UPDATE article SET image_url = 'javascript:x' WHERE id = 2")
    assert build_stories(conn, "m", NOW, "es")[0].image_url == "https://img.example/c.jpg"   # unsafe one skipped
