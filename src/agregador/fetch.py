import logging
import sqlite3
import time
from calendar import timegm
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import feedparser
import httpx

from .config import FeedConfig
from .db import RETENTION_DAYS
from .normalize import clean_text, normalize_url

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; agregador/0.1; personal news aggregator)"
TIMEOUT = httpx.Timeout(15.0)
FUTURE_WARN = timedelta(hours=6)  # clock skew of a feed (Daily Maverick: ~2 h) is normal; days ahead is not


@dataclass(frozen=True)
class Article:
    guid: str | None
    url: str
    title: str
    summary: str
    image_url: str | None
    published_at: datetime


def _entry_time(entry) -> datetime:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(timegm(parsed), UTC)
    return datetime.now(UTC)


def _entry_image(entry) -> str | None:
    for key in ("media_content", "media_thumbnail"):
        for media in entry.get(key, []) or []:
            if media.get("url"):
                return media["url"]
    for enc in entry.get("enclosures", []) or []:
        if enc.get("type", "").startswith("image/") and enc.get("href"):
            return enc["href"]
    return None


def parse_feed(content: bytes) -> list[Article]:
    parsed = feedparser.parse(content)
    articles = []
    for entry in parsed.entries:
        link = entry.get("link")
        title = clean_text(entry.get("title"))
        if not link or not title:
            continue
        articles.append(
            Article(
                guid=entry.get("id") or None,
                url=normalize_url(link),
                title=title,
                summary=clean_text(entry.get("summary")),
                image_url=_entry_image(entry),
                published_at=_entry_time(entry),
            )
        )
    return articles


def _upsert_feed(conn: sqlite3.Connection, feed: FeedConfig) -> sqlite3.Row:
    conn.execute(
        "INSERT INTO feed (name, url, country, paywall) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(url) DO UPDATE SET name = excluded.name, country = excluded.country, "
        "paywall = excluded.paywall",
        (feed.name, feed.url, feed.country, int(feed.paywall)),
    )
    return conn.execute("SELECT * FROM feed WHERE url = ?", (feed.url,)).fetchone()


def fetch_feed(conn: sqlite3.Connection, client: httpx.Client, feed: FeedConfig) -> int:
    """Fetch one feed and store new articles. Returns the number inserted."""
    row = _upsert_feed(conn, feed)
    headers = {"User-Agent": feed.user_agent} if feed.user_agent else {}
    if row["etag"]:
        headers["If-None-Match"] = row["etag"]
    if row["last_modified"]:
        headers["If-Modified-Since"] = row["last_modified"]
    now_dt = datetime.now(UTC)
    now = now_dt.isoformat()
    try:
        resp = client.get(feed.url, headers=headers)
        if resp.status_code == 304:
            conn.execute(
                "UPDATE feed SET last_success_at = ?, consecutive_failures = 0, "
                "last_error = NULL WHERE id = ?",
                (now, row["id"]),
            )
            conn.commit()
            return 0
        resp.raise_for_status()
        articles = parse_feed(resp.content)
    except Exception as exc:  # one bad feed must never abort the run
        log.warning("feed %s failed: %s", feed.name, exc)
        conn.execute(
            "UPDATE feed SET consecutive_failures = consecutive_failures + 1, "
            "last_error = ? WHERE id = ?",
            (str(exc)[:500], row["id"]),
        )
        conn.commit()
        return 0

    inserted = 0
    cutoff = datetime.now(UTC) - timedelta(days=RETENTION_DAYS)
    for a in articles:
        if a.published_at < cutoff:  # would be purged right away; don't churn it in and out
            continue
        published = a.published_at
        if published > now_dt:  # a feed with a wrong clock or a scheduled piece: never trust a date from the future
            if published - now_dt > FUTURE_WARN:
                log.warning("feed %s: %r is dated %s, in the future; stored as fetched now",
                            feed.name, a.title[:60], published.isoformat())
            published = now_dt
        cur = conn.execute(
            "INSERT OR IGNORE INTO article "
            "(feed_id, guid, url, title, summary, image_url, published_at, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row["id"], a.guid, a.url, a.title, a.summary, a.image_url,
             published.isoformat(), now),
        )
        inserted += cur.rowcount
    conn.execute(
        "UPDATE feed SET etag = ?, last_modified = ?, last_success_at = ?, "
        "consecutive_failures = 0, last_error = NULL WHERE id = ?",
        (resp.headers.get("etag"), resp.headers.get("last-modified"), now, row["id"]),
    )
    conn.commit()
    log.info("feed %s: %d articles, %d new", feed.name, len(articles), inserted)
    return inserted


def fetch_all(conn: sqlite3.Connection, feeds: list[FeedConfig]) -> int:
    total = 0
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
    ) as client:
        for feed in feeds:
            total += fetch_feed(conn, client, feed)
            time.sleep(0.2)
    return total
