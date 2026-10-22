"""Daily editions: a frozen snapshot of every tab, with its images and a manifest tying each image to its story.

An edition lives in `<site>/<YYYY-MM-DD>/`: `edition.json` (the data, written once), `img/` (thumbnails) and the
pages. The pages are a pure function of `edition.json`, so `refresh_site` can rebuild them all (new template,
new neighbours in the archive) without ever touching what was published. The home page is the latest edition.
"""
import json
import logging
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from .config import Country
from .fetch import USER_AGENT
from .images import save_story_image
from .render import (LABELS, Story, _DAYS_EN, _MONTHS_EN, _env, _web_url, _write_atomic, front_pages, render_page)

log = logging.getLogger(__name__)

EDITION_HOURS = 24  # an edition covers the day before it
IMAGE_RANKS = {0, 1, 3, 4, 5}  # positions on the page that show a picture (see index.html)
VERSION = 1


def edition_date(now: datetime, tz: str) -> date:
    return now.astimezone(ZoneInfo(tz)).date()


def _story_json(rank: int, story: Story, image: dict | None) -> dict:
    return {
        "rank": rank, "cluster_id": story.cluster_id, "headline": story.headline, "url": story.url,
        "source": story.source, "source_paywall": story.source_paywall, "updated_at": story.updated_at.isoformat(),
        "score": story.score, "involved": story.involved, "image": image,
        "articles": [{"outlet": s["name"], "title": s["title"], "url": s["url"],
                      "published_at": s["published_at"].isoformat(), "similarity": s["similarity"],
                      "image_url": s["image_url"], "paywall": s["paywall"]} for s in story.sources],
    }


def _story(j: dict) -> Story:
    sources = [{"name": a["outlet"], "title": a["title"], "url": a["url"], "similarity": a["similarity"],
                "published_at": datetime.fromisoformat(a["published_at"]), "image_url": a["image_url"],
                "paywall": a["paywall"]} for a in j["articles"]]
    return Story(
        cluster_id=j["cluster_id"], headline=j["headline"], url=j["url"], source=j["source"],
        source_paywall=j["source_paywall"], image_url=None, updated_at=datetime.fromisoformat(j["updated_at"]),
        score=j["score"], sources=sources, image=j["image"], fixed_involved=j["involved"],
    )


def build_edition(conn, model: str, now: datetime, countries: list[Country], tz: str, out_dir: Path,
                  client: httpx.Client | None = None) -> dict:
    """The edition's data; with a `client`, the thumbnails of the stories that show a picture are saved too."""
    day = edition_date(now, tz).isoformat()
    pages = front_pages(conn, model, now, countries, EDITION_HOURS)
    tabs = []
    for c in countries:
        stories = []
        for i, story in enumerate(pages[c.code]):
            image = None
            if client and i in IMAGE_RANKS:
                urls = {}  # image URL -> the article it came from, the representative's first
                for src in story.sources:
                    if url := _web_url(src["image_url"]):
                        urls.setdefault(url, {"url": url, "outlet": src["name"], "article_url": src["url"]})
                name = f"{c.code}-{i + 1:02d}-{story.cluster_id}.webp"
                image = save_story_image(client, list(urls.values()), story.headline, out_dir / day / "img" / name)
                if image:
                    image["file"] = f"img/{image['file']}"
            stories.append(_story_json(i + 1, story, image))
        tabs.append({"code": c.code, "name": c.name, "lang": c.lang, "tz": c.tz, "show_flags": c.show_flags,
                     "stories": stories})
    return {"version": VERSION, "date": day, "generated_at": now.isoformat(), "tz": tz,
            "window_hours": EDITION_HOURS, "tabs": tabs}


def make_edition(conn, model: str, countries: list[Country], out_dir: Path, tz: str, now: datetime | None = None,
                 force: bool = False, images: bool = True) -> Path | None:
    """Publish today's edition and rebuild the site. None if today's already exists (unless `force`)."""
    now = now or datetime.now(UTC)
    target = out_dir / edition_date(now, tz).isoformat()
    if (target / "edition.json").exists() and not force:
        return None
    shutil.rmtree(target / "img", ignore_errors=True)  # a forced rebuild must not leave orphan images
    client = None
    if images:
        client = httpx.Client(headers={"User-Agent": USER_AGENT})
    try:
        data = build_edition(conn, model, now, countries, tz, out_dir, client)
    finally:
        if client:
            client.close()
    _write_json(target / "edition.json", data)
    refresh_site(out_dir)
    return target


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_editions(out_dir: Path) -> list[dict]:
    """Every published edition, oldest first."""
    editions = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("????-??-??/edition.json"))]
    return sorted(editions, key=lambda e: e["date"])


def _tab_page(edition: dict, index: int, neighbours: dict, latest: bool) -> tuple[Path, str]:
    """(path relative to the site, HTML) of one tab of an edition, or of the home copy of the latest edition."""
    tabs = edition["tabs"]
    countries = [Country(t["code"], t["name"], t["tz"], t["lang"], show_flags=t["show_flags"]) for t in tabs]
    tab, first = tabs[index], index == 0
    sub = "" if first else f"{tab['code']}/"
    up = ("" if latest else "../") + ("" if first else "../")  # from this page to the site root

    def href(day: str | None) -> str | None:
        if not day:
            return None
        has = tab["code"] in neighbours["tabs"][day]
        return f"{up}{day}/" + (f"{tab['code']}/" if has and not first else "") + "index.html"

    links = {"assets": f"{up}{edition['date']}/" if latest else ("" if first else "../"),
             "prev": href(neighbours["prev"]), "next": href(neighbours["next"]) if not latest else None,
             "archive": f"{up}archive/index.html"}
    html = render_page([_story(j) for j in tab["stories"]], datetime.fromisoformat(edition["generated_at"]),
                       countries[index], countries, root="" if first else "../", edition=links)
    where = Path(sub) if latest else Path(edition["date"]) / sub
    return where / "index.html", html


def _archive_page(editions: list[dict]) -> str:
    months: dict[str, list] = {}
    for e in reversed(editions):
        d = date.fromisoformat(e["date"])
        months.setdefault(f"{_MONTHS_EN[d.month - 1]} {d.year}", []).append({
            "date": e["date"], "label": f"{_DAYS_EN[d.weekday()]} {d.day}",
            "tabs": [{"code": t["code"], "name": t["name"]} for t in e["tabs"]]})
    return _env.get_template("archive.html").render(
        L=LABELS["en"], root="../", months=[{"name": n, "days": days} for n, days in months.items()])


def refresh_site(out_dir: Path) -> None:
    """Rebuild every page from the stored editions: each edition, the archive and the home page (the latest)."""
    editions = load_editions(out_dir)
    if not editions:
        return
    days = [e["date"] for e in editions]
    tabs_of = {e["date"]: [t["code"] for t in e["tabs"]] for e in editions}
    for i, e in enumerate(editions):
        neighbours = {"prev": days[i - 1] if i else None, "next": days[i + 1] if i + 1 < len(days) else None,
                      "tabs": tabs_of}
        for t in range(len(e["tabs"])):
            path, html = _tab_page(e, t, neighbours, latest=False)
            _write_atomic(out_dir / path, html)
        if i == len(editions) - 1:
            for t in range(len(e["tabs"])):
                path, html = _tab_page(e, t, neighbours, latest=True)
                _write_atomic(out_dir / path, html)
    _write_atomic(out_dir / "archive" / "index.html", _archive_page(editions))
