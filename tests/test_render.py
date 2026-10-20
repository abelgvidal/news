from datetime import UTC, datetime, timedelta

from agregador.render import Story, render_page, score

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def story(cid, outlets, hours_ago, headline="Titular <b>"):
    t = NOW - timedelta(hours=hours_ago)
    sources = [{"name": f"M{i}", "title": headline, "url": f"https://e.com/{cid}/{i}", "published_at": t}
               for i in range(outlets)]
    return Story(cid, headline, sources[0]["url"], None, t, score(outlets, t, NOW), sources)


def test_score_prefers_more_outlets_and_fresher():
    assert score(5, NOW, NOW) > score(3, NOW, NOW)
    assert score(3, NOW, NOW) > score(3, NOW - timedelta(hours=24), NOW)


def test_page_shows_only_top_ten_with_outlet_names_and_no_links():
    stories = [story(i, 2 + i % 5, i) for i in range(1, 16)] + [story(99, 1, 0)]
    html = render_page(stories, NOW)
    assert html.count('<article class="story') == 10
    assert "<b>Cubierto por</b>" in html and "M0" in html and "M1" in html


def test_page_escapes_html_and_shows_summary():
    s = story(1, 3, 1)
    s.summary = "Resumen <script>"
    html = render_page([s], NOW)
    assert "&lt;b&gt;" in html and "Titular <b>" not in html
    assert "&lt;script&gt;" in html


def test_tabs_link_relative_to_each_page():
    from agregador.config import Country
    es, ar = Country("es", "España", "Europe/Madrid"), Country("ar", "Argentina", "America/Argentina/Buenos_Aires")
    home = render_page([story(1, 3, 1)], NOW, es, [es, ar], root="")
    sub = render_page([story(1, 3, 1)], NOW, ar, [es, ar], root="../")
    assert 'href="index.html"' in home and 'href="ar/index.html"' in home
    assert 'href="../index.html"' in sub and 'href="../ar/index.html"' in sub
    assert 'aria-current="page">Argentina' in sub and "Argentina" in sub


def test_single_country_has_no_tabs():
    assert "<nav" not in render_page([story(1, 3, 1)], NOW)


def test_outlets_link_to_their_own_article_with_the_rest_in_a_details_toggle():
    s = story(1, 7, 1)
    html = render_page([s], NOW)
    assert 'href="https://e.com/1/0"' in html and 'rel="noopener noreferrer nofollow"' in html
    assert html.count("<details") == 1 and "y 3 más" in html   # 4 shown, 3 folded
    folded = html[html.index("<details"):]
    shown = html[:html.index("<details")]
    assert all(f'href="https://e.com/1/{i}"' in folded for i in (4, 5, 6))
    assert all(f'href="https://e.com/1/{i}"' in shown for i in (0, 1, 2, 3))
    assert 'href="https://e.com/1/4"' not in shown


def test_few_outlets_need_no_toggle():
    assert "<details" not in render_page([story(1, 3, 1)], NOW)


def test_only_http_urls_become_links_and_unsafe_ones_are_dropped():
    s = story(1, 3, 1)
    s.sources[0]["url"] = "javascript:alert(1)"
    s.sources[1]["url"] = "data:text/html,<script>"
    html = render_page([s], NOW)
    assert "javascript:" not in html and "data:text" not in html
    assert "M0" in html and "M1" in html  # still listed, just not linked
    assert 'href="https://e.com/1/2"' in html


def test_only_the_outlets_that_cover_the_event_are_counted_and_linked():
    s = story(1, 6, 1)
    s.covering = ["M0", "M2", "M4"]
    assert (s.outlets, s.outlet_names) == (3, ["M0", "M2", "M4"])
    html = render_page([s], NOW)
    assert "3 medios" in html and 'href="https://e.com/1/2"' in html
    assert 'href="https://e.com/1/1"' not in html and 'href="https://e.com/1/3"' not in html


def test_ranking_uses_the_honest_count_and_skipped_stories_never_appear(tmp_path):
    from agregador import db
    from agregador.render import attach_summaries, top_stories

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    for cid in (1, 2, 3):
        conn.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at) "
                     "VALUES (?, 'm', x'00', 3, 'x', 'x')", (cid,))
    rows = [(1, "ok", '["M0","M1"]', "Cubre dos"), (2, "skip", "[]", ""), (3, "ok", '["M0","M1","M2","M3"]', "Cubre cuatro")]
    for cid, status, outlets, headline in rows:
        conn.execute("INSERT INTO summary (cluster_id, model, n_members, headline, text, created_at, outlets, status) "
                     "VALUES (?, 'x', 6, ?, 'a\nb', 'x', ?, ?)", (cid, headline, outlets, status))
    stories = [story(1, 6, 1), story(2, 6, 1), story(3, 6, 1)]   # raw counts: six outlets each
    attach_summaries(conn, stories, NOW)
    top = top_stories(stories)
    assert [s.headline for s in top] == ["Cubre cuatro", "Cubre dos"]   # 4 outlets beats 2; the skipped one is gone
    assert [s.outlets for s in top] == [4, 2]


# --- image chosen by the final summary -------------------------------------------------------

def _image_db(tmp_path):
    import numpy as np

    from agregador import db

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    conn.execute("INSERT INTO feed (id, name, url) VALUES (1, 'A', 'a')")
    conn.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at) "
                 "VALUES (1, 'm', x'00', 3, 'x', 'x')")
    vecs = {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.6, 0.8]}  # article 1 is the one about the story
    for aid, v in vecs.items():
        conn.execute("INSERT INTO article (id, feed_id, url, title, published_at, fetched_at) "
                     "VALUES (?, 1, ?, 't', 'x', 'x')", (aid, f"https://e.com/{aid}"))
        conn.execute("INSERT INTO embedding (article_id, model, vector) VALUES (?, 'm', ?)",
                     (aid, np.array(v, dtype=np.float32).tobytes()))
    conn.execute("INSERT INTO summary (cluster_id, model, n_members, headline, text, created_at, outlets, status) "
                 "VALUES (1, 'x', 3, 'Titular final', 'a\nb', 'x', '[]', 'ok')")
    return conn


def _story_with_images():
    s = story(1, 3, 1)
    for i, src in enumerate(s.sources, 1):
        src["article_id"], src["image_url"] = i, f"https://img.example/{i}.jpg"
    s.sources[0]["name"], s.sources[1]["name"], s.sources[2]["name"] = "M0", "M1", "M2"
    return s


def test_the_image_comes_from_the_article_closest_to_the_final_summary(tmp_path, monkeypatch):
    import numpy as np

    from agregador.render import attach_summaries

    conn = _image_db(tmp_path)
    monkeypatch.setattr("agregador.embed.encode", lambda texts, model: np.array([[0.0, 1.0]], dtype=np.float32))
    s = _story_with_images()
    s.image_url = "https://img.example/1.jpg"          # what the cluster centroid would have picked
    attach_summaries(conn, [s], NOW, "m")
    assert s.image_url == "https://img.example/2.jpg"  # article 2 matches the summary vector


def test_the_chosen_image_is_cached_and_recomputed_only_for_a_new_summary(tmp_path, monkeypatch):
    import numpy as np

    from agregador.render import attach_summaries

    conn = _image_db(tmp_path)
    calls = []
    monkeypatch.setattr("agregador.embed.encode",
                        lambda texts, model: calls.append(1) or np.array([[0.0, 1.0]], dtype=np.float32))
    attach_summaries(conn, [_story_with_images()], NOW, "m")
    attach_summaries(conn, [_story_with_images()], NOW, "m")
    assert len(calls) == 1 and conn.execute("SELECT image_url FROM summary").fetchone()[0].endswith("2.jpg")
    conn.execute("UPDATE summary SET image_url = NULL")    # what a regenerated summary starts with
    attach_summaries(conn, [_story_with_images()], NOW, "m")
    assert len(calls) == 2


def test_only_articles_that_cover_the_event_can_lend_their_image(tmp_path, monkeypatch):
    import numpy as np

    from agregador.render import attach_summaries

    conn = _image_db(tmp_path)
    monkeypatch.setattr("agregador.embed.encode", lambda texts, model: np.array([[0.0, 1.0]], dtype=np.float32))
    conn.execute("UPDATE summary SET outlets = '[\"M0\", \"M2\"]'")   # M1, the best match, does not cover it
    s = _story_with_images()
    attach_summaries(conn, [s], NOW, "m")
    assert s.image_url == "https://img.example/3.jpg"                     # best among M0 and M2


def test_a_story_without_any_image_keeps_none_and_does_not_break(tmp_path, monkeypatch):
    from agregador.render import attach_summaries

    conn = _image_db(tmp_path)
    monkeypatch.setattr("agregador.embed.encode", lambda *a: (_ for _ in ()).throw(AssertionError("not needed")))
    s = story(1, 3, 1)
    s.image_url = None
    attach_summaries(conn, [s], NOW, "m")
    assert s.image_url is None and conn.execute("SELECT image_url FROM summary").fetchone()[0] == ""
