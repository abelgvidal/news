import json
from datetime import UTC, datetime, timedelta

from agregador.config import Country
from agregador.edition import build_edition, edition_date, load_editions, make_edition, refresh_site
from test_render import NOW, _db_with_cluster, _u

ES = Country("es", "Spain", "Europe/Madrid", "es")
INT = Country("int", "International", "Europe/Madrid", "en", show_flags=True)


def _conn(tmp_path):
    return _db_with_cluster(tmp_path, [
        ("A", "Titular central", _u(1, 0.02), "https://img.example/a.jpg", 30),
        ("B", "Titular parecido", _u(1, 0.0), None, 20),
    ])


def test_the_edition_belongs_to_the_local_day():
    late = datetime(2026, 10, 19, 23, 30, tzinfo=UTC)                   # already the 20th in Madrid
    assert edition_date(late, "Europe/Madrid").isoformat() == "2026-10-20"
    assert edition_date(late, "UTC").isoformat() == "2026-10-19"


def test_the_manifest_keeps_the_story_with_all_its_articles(tmp_path):
    data = build_edition(_conn(tmp_path), "m", NOW, [ES], "Europe/Madrid", tmp_path / "site")
    assert data["date"] == "2026-10-19" and data["version"] == 1
    (story,) = data["tabs"][0]["stories"]
    assert story["headline"] == "Titular central" and story["rank"] == 1 and story["image"] is None  # no client
    assert [a["outlet"] for a in story["articles"]] == ["A", "B"]
    json.dumps(data)                                                     # plain data


def test_an_edition_is_published_once_with_pages_archive_and_home(tmp_path):
    site = tmp_path / "site"
    conn = _conn(tmp_path)
    target = make_edition(conn, "m", [INT, ES], site, "Europe/Madrid", now=NOW, images=False)
    assert target == site / "2026-10-19" and (target / "edition.json").exists()
    for page in ("index.html", "es/index.html"):
        assert (target / page).exists() and (site / page).exists()      # the edition and the home copy
    assert (site / "archive" / "index.html").exists()
    assert make_edition(conn, "m", [INT, ES], site, "Europe/Madrid", now=NOW, images=False) is None
    assert make_edition(conn, "m", [INT, ES], site, "Europe/Madrid", now=NOW, images=False, force=True)


def test_the_home_is_the_latest_edition_and_editions_link_to_their_neighbours(tmp_path):
    site = tmp_path / "site"
    conn = _conn(tmp_path)
    for days in (0, 1, 2):
        make_edition(conn, "m", [INT, ES], site, "Europe/Madrid", now=NOW + timedelta(days=days), images=False)
    first, middle = (site / "2026-10-19" / "index.html").read_text(), (site / "2026-10-20" / "index.html").read_text()
    home, es_home = (site / "index.html").read_text(), (site / "es" / "index.html").read_text()
    assert 'rel="next"' in first and 'href="../2026-10-20/index.html"' in first and 'rel="prev"' not in first
    assert 'href="../2026-10-19/index.html"' in middle and 'href="../2026-10-21/index.html"' in middle
    assert 'rel="next"' not in home and 'href="2026-10-20/index.html"' in home   # the home links back, not forward
    assert 'href="../2026-10-20/es/index.html"' in es_home                        # keeping the tab
    assert "Friday 23" not in home and "Wednesday, 21 October 2026" in home


def test_the_archive_lists_every_edition_newest_first(tmp_path):
    site = tmp_path / "site"
    conn = _conn(tmp_path)
    for days in (0, 1):
        make_edition(conn, "m", [INT, ES], site, "Europe/Madrid", now=NOW + timedelta(days=days), images=False)
    html = (site / "archive" / "index.html").read_text()
    assert "October 2026" in html
    assert html.index("2026-10-20/index.html") < html.index("2026-10-19/index.html")
    assert "../2026-10-20/es/index.html" in html


def test_refresh_site_rebuilds_pages_from_the_manifest_alone(tmp_path):
    site = tmp_path / "site"
    make_edition(_conn(tmp_path), "m", [INT, ES], site, "Europe/Madrid", now=NOW, images=False)
    page = site / "2026-10-19" / "index.html"
    page.unlink()
    (site / "index.html").unlink()
    refresh_site(site)
    assert page.exists() and (site / "index.html").exists() and len(load_editions(site)) == 1


def test_images_are_referenced_relative_to_each_page(tmp_path):
    site = tmp_path / "site"
    make_edition(_conn(tmp_path), "m", [INT, ES], site, "Europe/Madrid", now=NOW, images=False)
    path = site / "2026-10-19" / "edition.json"
    data = json.loads(path.read_text())
    data["tabs"][0]["stories"] = json.loads(json.dumps(data["tabs"][1]["stories"]))            # the fixture only has Spanish stories
    for tab in data["tabs"]:
        tab["stories"][0]["image"] = {"file": f"img/{tab['code']}-01.webp", "width": 800, "height": 450, "sha256": "x",
                                      "original_url": "https://i.example/a.jpg", "outlet": "A",
                                      "article_url": "https://a.example/1"}
    path.write_text(json.dumps(data))
    refresh_site(site)
    assert 'src="img/int-01.webp"' in (site / "2026-10-19" / "index.html").read_text()
    assert 'src="../img/es-01.webp"' in (site / "2026-10-19" / "es" / "index.html").read_text()
    assert 'src="2026-10-19/img/int-01.webp"' in (site / "index.html").read_text()
    assert 'src="../2026-10-19/img/es-01.webp"' in (site / "es" / "index.html").read_text()
    assert 'alt="Titular central"' in (site / "index.html").read_text() and "Image: A" in (site / "index.html").read_text()
