from pathlib import Path

from agregador import db
from agregador.fetch import parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "sample.xml"


def test_parse_feed_skips_entries_without_title():
    articles = parse_feed(FIXTURE.read_bytes())
    assert len(articles) == 1
    a = articles[0]
    assert a.url == "https://example.com/noticia/1?id=7"
    assert a.summary == "Resumen con HTML ."
    assert a.image_url == "https://example.com/img.jpg"
    assert a.published_at.year == 2026


def test_migrate_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    v1 = db.migrate(conn)
    v2 = db.migrate(conn)
    assert v1 == v2 == len(list(db.MIGRATIONS_DIR.glob('*.sql')))
