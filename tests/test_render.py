from datetime import UTC, datetime, timedelta

import numpy as np

from agregador import db
from agregador.config import Country
from agregador.render import Story, build_stories, render_page, score, top_stories

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def story(cid, outlets, hours_ago, headline="Titular <b>", paywalled=()):
    t = NOW - timedelta(hours=hours_ago)
    sources = [{"name": f"M{i}", "title": headline if i == 0 else f"otro {i}", "url": f"https://e.com/{cid}/{i}",
                "published_at": t, "similarity": 1.0 - i * 0.01, "image_url": None, "paywall": f"M{i}" in paywalled}
               for i in range(outlets)]
    return Story(cid, headline, sources[0]["url"], "M0", "M0" in paywalled, None, t, score(outlets, t, NOW), sources)


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
    assert '<span class="source">M0</span> <span class="more">y 2 más</span>' in html


def test_html_in_a_headline_is_escaped():
    html = render_page([story(1, 3, 1)], NOW)
    assert "&lt;b&gt;" in html and "Titular <b>" not in html


def test_other_outlets_exclude_the_representative_and_sit_behind_one_toggle():
    s = story(1, 7, 1)                       # M0 is the representative; M1..M6 are the others
    assert [o["name"] for o in s.other_outlets] == ["M1", "M2", "M3", "M4", "M5", "M6"]
    html = render_page([s], NOW)
    assert html.count('<details class="byline"') == 1 and "y 6 más" in html
    shown, listed = html[:html.index("<details")], html[html.index('<div class="outlets">'):]
    assert 'href="https://e.com/1/0"' in shown                          # the headline itself
    assert all(f'href="https://e.com/1/{i}"' in listed for i in range(1, 7))   # every other outlet stays linked


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


def test_only_the_articles_inside_the_window_make_the_story(tmp_path):
    conn = _db_with_cluster(tmp_path, [
        ("A", "Titular viejo", _u(1, 0.0), None, 60 * 30),               # 30 h ago: outside a 24 h edition
        ("B", "Titular central", _u(1, 0.02), None, 30),
        ("C", "Titular cercano", _u(1, 0.04), None, 10),
    ])
    (s,) = build_stories(conn, "m", NOW, "es", 24)
    assert s.outlets == 2 and sorted(x["name"] for x in s.sources) == ["B", "C"]
    assert build_stories(conn, "m", NOW, "es", 48)[0].outlets == 3      # the live preview keeps its 48 h


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


# --- language per page, site title, locks ---------------------------------------------------

EN = Country("int", "International", "Europe/Madrid", "en")


def test_an_english_page_has_english_labels_dates_and_title():
    es = Country("es", "Spain", "Europe/Madrid", "es")
    html = render_page([story(1, 7, 1)], NOW, EN, [EN, es])
    assert '<html lang="en">' in html and "<title>The Daily Ten · International</title>" in html
    assert "<h1>The Daily Ten</h1>" in html
    assert "Monday, 19 October 2026" in html and "Most covered in the press" in html
    assert "and 6 more" in html and "Also in" not in html
    assert ">International</a>" in html and ">Spain</a>" in html            # tabs in English


def test_a_spanish_page_keeps_spanish_labels_but_the_english_title():
    html = render_page([story(1, 7, 1)], NOW)
    assert "<h1>The Daily Ten</h1>" in html and "lunes, 19 de octubre de 2026" in html
    assert "y 6 más" in html and "También en" not in html


def test_paywalled_outlets_get_a_lock_where_they_are_named_and_others_do_not():
    s = story(1, 4, 1, paywalled={"M0", "M2"})
    html = render_page([s], NOW, EN)
    assert html.count("🔒") == 2        # the source label, and M2 in the list behind "and 2 more"
    assert 'title="May require a subscription"' in html
    assert s.other_outlets[1]["paywall"] is True and s.other_outlets[0]["paywall"] is False


def test_no_lock_when_nothing_is_paywalled():
    assert "🔒" not in render_page([story(1, 4, 1)], NOW, EN)


# --- keeping world news out of an edition -----------------------------------------------------

def test_excluded_stories_are_skipped_and_the_next_ones_fill_the_front_page():
    stories = [story(i, 9 - i, 1) for i in range(1, 9)]       # 8..1 outlets: the last one never ranks
    top = top_stories(stories, exclude=lambda s: s.cluster_id in (1, 2))
    assert [s.cluster_id for s in top] == [3, 4, 5, 6, 7]


def test_write_site_keeps_world_news_out_of_spain_but_in_international(tmp_path):
    from agregador.render import write_site

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    OTHER = {"int": ("Foreign A", "Foreign B"), "es": ("Spanish A", "Spanish B")}

    def add_cluster(cid, country, headline, section, vec):
        conn.execute("INSERT INTO cluster (id, model, vector_sum, size, created_at, last_published_at, country) "
                     "VALUES (?, 'm', x'00', 2, 'x', ?, ?)", (cid, NOW.isoformat(), country))
        for n, outlet in enumerate(OTHER[country]):
            conn.execute("INSERT OR IGNORE INTO feed (name, url, country) VALUES (?, ?, ?)", (outlet, f"f-{outlet}", country))
            fid = conn.execute("SELECT id FROM feed WHERE name = ?", (outlet,)).fetchone()[0]
            aid = cid * 10 + n
            conn.execute("INSERT INTO article (id, feed_id, url, title, published_at, fetched_at) VALUES (?, ?, ?, ?, ?, 'x')",
                         (aid, fid, f"https://{outlet.replace(' ', '')}.example/{section}/{aid}", headline,
                          (NOW - timedelta(minutes=n)).isoformat()))
            conn.execute("INSERT INTO embedding (article_id, model, vector) VALUES (?, 'm', ?)",
                         (aid, np.array(vec, dtype=np.float32).tobytes()))
            conn.execute("INSERT INTO cluster_member (article_id, cluster_id) VALUES (?, ?)", (aid, cid))

    add_cluster(1, "int", "Trump meets Putin and Xi in China", "world", [1, 0.02])
    add_cluster(2, "es", "Trump se reúne con Putin y Xi en China", "espana", [1, 0.03])          # same story, no section hint
    add_cluster(3, "es", "Sánchez disuelve las Cortes y convoca elecciones", "espana", [0.1, 1])
    add_cluster(4, "es", "Meloni envía militares a Milán", "internacional", [0.7, -0.7])          # world by URL section
    int_c, es_c = Country("int", "International", "Europe/Madrid", "en"), Country("es", "Spain", "Europe/Madrid", "es", True)
    write_site(conn, "m", tmp_path / "site", [int_c, es_c], now=NOW)
    spain = (tmp_path / "site" / "es" / "index.html").read_text()
    assert "Sánchez disuelve las Cortes" in spain
    assert "Meloni envía militares" not in spain                       # excluded by the URL section
    assert "Trump se reúne con Putin" not in spain                      # excluded: it is already in International
    assert "Trump meets Putin and Xi in China" in (tmp_path / "site" / "index.html").read_text()


# --- at most N stories per country --------------------------------------------------------------

def _about(cid, outlets, headline):
    return story(cid, outlets, 1, headline=headline)


def test_one_story_per_country_keeps_the_best_ranked_and_lets_the_next_ones_in():
    stories = [
        _about(1, 9, "Alberta voters reject separation from Canada"),
        _about(2, 8, "France faces reckoning with policing after riots in Paris"),
        _about(3, 8, "France imposes curfews in Paris and other cities"),
        _about(4, 7, "Putin floats a summit with Trump and Xi to discuss Ukraine deal"),     # mixed: never limited
        _about(5, 6, "Four Canadian service members killed in Alberta base fire"),
        _about(6, 5, "Limoges town hall burns in French protests"),
        _about(7, 4, "Japan launches mission to Mars moons from Tokyo"),
    ]
    assert [s.cluster_id for s in top_stories(stories, max_per_country=1)] == [1, 2, 4, 7]
    assert [s.cluster_id for s in top_stories(stories, max_per_country=2)] == [1, 2, 3, 4, 5, 7]
    assert len(top_stories(stories)) == 7                                              # no limit by default


def test_the_cap_and_the_exclusion_work_together():
    stories = [_about(1, 9, "France imposes curfews in Paris"), _about(2, 8, "France faces riots in Paris"),
               _about(3, 7, "Japan launches mission from Tokyo")]
    top = top_stories(stories, exclude=lambda s: s.cluster_id == 1, max_per_country=1)
    assert [s.cluster_id for s in top] == [2, 3]


# --- flags ----------------------------------------------------------------------------------------

def test_the_page_shows_no_flag_and_no_count_and_time_line_even_if_the_edition_asks_for_flags():
    flags_on = Country("int", "International", "Europe/Madrid", "en", show_flags=True)
    html = render_page([story(1, 7, 1)], NOW, flags_on)
    assert 'class="flag"' not in html and "kicker" not in html and "7 outlets" not in html


def test_an_edition_without_flags_shows_none():
    html = render_page([_about(1, 5, "France imposes curfews in Paris")], NOW, Country("es", "Spain", "Europe/Madrid", "es"))
    assert 'class="flag"' not in html and "🇫🇷" not in html


def test_an_unusually_long_headline_is_replaced_but_the_story_and_its_outlet_count_do_not_change(tmp_path):
    conn = _db_with_cluster(tmp_path, [
        ("A", "Detenido un sospechoso tras el incendio del refugio de montaña en Baviera por una presunta negligencia grave", _u(1, 0.02), None, 50),
        ("B", "Detenido un sospechoso tras el incendio del refugio", _u(1, 0.0), None, 40),
        ("C", "Incendio del refugio: detenido un sospechoso", _u(1, 0.05), None, 30),
        ("D", "Detienen a un sospechoso por el incendio del refugio", _u(1, 0.08), None, 20),
        ("E", "Un detenido por el incendio en el refugio", _u(1, -0.03), None, 10),
    ])
    (s,) = build_stories(conn, "m", NOW, "es")
    assert (s.headline, s.source) == ("Detenido un sospechoso tras el incendio del refugio", "B")
    assert s.outlets == 5 and abs(s.sources[0]["similarity"] - 1.0) < 1e-6
    assert "A" in [o["name"] for o in s.other_outlets] and "B" not in [o["name"] for o in s.other_outlets]
