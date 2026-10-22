"""Probe feeds: does each answer, how many items, how deep (hours), how fresh, what is in it.

    uv run python scripts/probe_feeds.py                      # the feeds in production (feeds.toml)
    uv run python scripts/probe_feeds.py --url https://example.com/feed   # a one-off, before adding it
    uv run python scripts/probe_feeds.py --only Africanews

Run it from the server too: outlets block datacenter IPs that they do not block at home.
It only reads.
"""
import argparse
import calendar
import sys
import time
import tomllib
from pathlib import Path

import feedparser
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agregador.fetch import USER_AGENT  # noqa: E402

TIMEOUT = 20.0


def probe(name: str, url: str, user_agent: str | None = None) -> str:
    try:
        r = httpx.get(url, headers={"User-Agent": user_agent or USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as e:
        return f"{name:<26} ERR  {str(e)[:70]}"
    if r.status_code != 200:
        return f"{name:<26} {r.status_code}  blocked or wrong URL"
    entries = feedparser.parse(r.content).entries
    if not entries:
        return f"{name:<26} 200  EMPTY feed"
    now = time.time()
    ts = [calendar.timegm(e.published_parsed) for e in entries if getattr(e, "published_parsed", None)]
    images = sum(1 for e in entries if e.get("media_content") or e.get("media_thumbnail") or e.get("enclosures"))
    span = f"{(now - min(ts)) / 3600:6.0f} h deep, newest {(now - max(ts)) / 3600:5.1f} h" if ts else "no dates"
    return f"{name:<26} 200  {len(entries):>3} items, {span}, images {images}/{len(entries)} | {entries[0].title[:50]}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, default=Path(__file__).resolve().parents[1] / "feeds.toml")
    p.add_argument("--url", help="probe one URL instead of a file")
    p.add_argument("--only", help="substring of an outlet's name")
    args = p.parse_args()
    if args.url:
        print(probe("(url)", args.url))
        return
    data = tomllib.loads(args.file.read_text(encoding="utf-8"))
    for entry in data.get("feed", []):
        if args.only and args.only.lower() not in entry["name"].lower():
            continue
        print(probe(entry["name"], entry["url"], entry.get("user_agent")))


if __name__ == "__main__":
    main()
