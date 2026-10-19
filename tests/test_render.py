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
    assert "<b>Cubierto por</b> M0, M1" in html
    assert "<a " not in html and "https://e.com" not in html  # no links to the outlets


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
