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


def test_feed_user_agent_override_is_sent(tmp_path):
    import httpx

    from agregador.config import FeedConfig
    from agregador.fetch import fetch_feed

    seen = {}

    def handler(request):
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, content=FIXTURE.read_bytes())

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    client = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "default-agent"})
    fetch_feed(conn, client, FeedConfig("A", "https://a.example/rss"))
    assert seen["ua"] == "default-agent"
    fetch_feed(conn, client, FeedConfig("B", "https://b.example/rss", user_agent="custom-agent"))
    assert seen["ua"] == "custom-agent"


def test_the_paywall_flag_is_stored_and_updated_from_the_config(tmp_path):
    import httpx

    from agregador.config import FeedConfig
    from agregador.fetch import fetch_feed

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=FIXTURE.read_bytes())))
    fetch_feed(conn, client, FeedConfig("A", "https://a.example/rss", paywall=True))
    assert conn.execute("SELECT paywall FROM feed").fetchone()[0] == 1
    fetch_feed(conn, client, FeedConfig("A", "https://a.example/rss", paywall=False))      # edited in feeds.toml
    assert conn.execute("SELECT paywall FROM feed").fetchone()[0] == 0


def _rss(pub_date: str) -> bytes:
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
        f"<item><title>Halloween plans</title><link>https://a.example/h</link><guid>h</guid><pubDate>{pub_date}</pubDate></item>"
        "</channel></rss>"
    ).encode()


def test_an_article_dated_in_the_future_is_stored_with_the_fetch_time(tmp_path):
    import httpx
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from agregador.config import FeedConfig
    from agregador.fetch import fetch_feed

    future = datetime.now(UTC) + timedelta(days=4)
    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=_rss(format_datetime(future)))))
    fetch_feed(conn, client, FeedConfig("A", "https://a.example/rss"))
    row = conn.execute("SELECT published_at, fetched_at FROM article").fetchone()
    assert row["published_at"] == row["fetched_at"]
    # seen again later with the same future date: ignored as a duplicate, the stored date does not move
    fetch_feed(conn, client, FeedConfig("A", "https://a.example/rss"))
    assert conn.execute("SELECT COUNT(*) FROM article").fetchone()[0] == 1
    assert conn.execute("SELECT published_at FROM article").fetchone()[0] == row["published_at"]


def test_an_article_dated_in_the_past_keeps_its_date(tmp_path):
    import httpx
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from agregador.config import FeedConfig
    from agregador.fetch import fetch_feed

    past = (datetime.now(UTC) - timedelta(hours=5)).replace(microsecond=0)
    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=_rss(format_datetime(past)))))
    fetch_feed(conn, client, FeedConfig("A", "https://a.example/rss"))
    assert conn.execute("SELECT published_at FROM article").fetchone()[0] == past.isoformat()


def test_feed_health_ignores_feeds_dropped_from_the_config(tmp_path):
    from agregador.cli import feed_health
    from agregador.config import FeedConfig

    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    for name, url, fails in [("Kept", "https://a.example/rss", 0), ("Retired", "https://b.example/rss", 49)]:
        conn.execute("INSERT INTO feed (name, url, consecutive_failures) VALUES (?, ?, ?)", (name, url, fails))

    rows, retired = feed_health(conn, [FeedConfig(name="Kept", url="https://a.example/rss")])

    assert [r["name"] for r in rows] == ["Kept"]
    assert retired == 1
