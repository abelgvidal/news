"""Render the front page as static HTML: each story is a cluster's most representative real headline."""
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape

from .config import Country
from .foreign import centroid, entity_stems, judge
from .representative import choose_story

ACTIVE_HOURS = 48
HALF_LIFE_HOURS = 24
MAX_STORIES = 10
OTHERS_TOP = 40  # stories of the other editions a story is compared with

DEFAULT_COUNTRY = Country("es", "España", "Europe/Madrid")
_DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"]

_env = Environment(loader=PackageLoader("agregador"), autoescape=select_autoescape())


def fecha_es(dt: datetime, tz: str) -> str:
    d = dt.astimezone(ZoneInfo(tz))
    return f"{_DAYS[d.weekday()]}, {d.day} de {_MONTHS[d.month - 1]} de {d.year}"


_DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
LABELS = {
    "es": {"site": "News Aggregator", "outlets": "medios", "also_in": "También en", "more": "y {n} más",
           "dateline": "Lo más citado en los medios · actualizado a las", "paywall": "Puede requerir suscripción",
           "empty": "Todavía no hay noticias agrupadas."},
    "en": {"site": "News Aggregator", "outlets": "outlets", "also_in": "Also in", "more": "and {n} more",
           "dateline": "Most covered in the press · updated at", "paywall": "May require a subscription",
           "empty": "No grouped stories yet."},
}


def fecha_en(dt: datetime, tz: str) -> str:
    d = dt.astimezone(ZoneInfo(tz))
    return f"{_DAYS_EN[d.weekday()]}, {d.day} {_MONTHS_EN[d.month - 1]} {d.year}"


def hora_local(dt: datetime, tz: str) -> str:
    return dt.astimezone(ZoneInfo(tz)).strftime("%H:%M")


def _web_url(url: str | None) -> str | None:
    """Feed content is untrusted: only http(s) URLs may become links or images."""
    return url if url and urlsplit(url).scheme in ("http", "https") else None


@dataclass
class Story:
    cluster_id: int
    headline: str  # the representative article's own headline, as published
    url: str | None  # that article
    source: str  # its outlet
    source_paywall: bool  # that outlet's articles often need a subscription
    image_url: str | None
    updated_at: datetime
    score: float
    sources: list[dict]  # every article, the representative first, then by similarity to it
    centroid: np.ndarray | None = None  # mean embedding of the story's articles
    entities: frozenset = frozenset()  # proper-noun stems of its headlines

    @property
    def outlets(self) -> int:
        return len({s["name"] for s in self.sources})

    @property
    def other_outlets(self) -> list[dict]:
        """The other outlets, each with the article closest to the representative one.

        An outlet without a safe URL is still listed, just not linked.
        """
        links: dict[str, str | None] = {}
        for src in self.sources:
            if src["name"] == self.source:
                continue
            url = _web_url(src["url"])
            if links.get(src["name"]) is None:
                links[src["name"]] = url
        lock = {src["name"]: src["paywall"] for src in self.sources}
        return [{"name": name, "url": url, "paywall": lock[name]} for name, url in links.items()]


def score(outlets: int, last_published: datetime, now: datetime) -> float:
    """Distinct outlets, halved for every HALF_LIFE_HOURS since the latest article."""
    age_h = max((now - last_published).total_seconds() / 3600, 0)
    return outlets * math.pow(0.5, age_h / HALF_LIFE_HOURS)


def build_stories(
    conn: sqlite3.Connection, model: str, now: datetime, country: str = "es"
) -> list[Story]:
    since = (now - timedelta(hours=ACTIVE_HOURS)).isoformat()
    clusters = conn.execute(
        "SELECT id, last_published_at FROM cluster "
        "WHERE model = ? AND country = ? AND last_published_at >= ?", (model, country, since)
    ).fetchall()
    stories = []
    for c in clusters:
        members = conn.execute(
            "SELECT a.title, a.url, a.image_url, a.published_at, f.name, f.paywall, e.vector "
            "FROM cluster_member m JOIN article a ON a.id = m.article_id "
            "JOIN feed f ON f.id = a.feed_id JOIN embedding e ON e.article_id = a.id AND e.model = ? "
            "WHERE m.cluster_id = ?", (model, c["id"])
        ).fetchall()
        if not members:
            continue
        vecs = np.stack([np.frombuffer(m["vector"], dtype=np.float32) for m in members])
        published = [datetime.fromisoformat(m["published_at"]) for m in members]
        titles = [m["title"] for m in members]
        best, core = choose_story(titles, vecs, published)  # the cluster may hold neighbouring topics: keep the story
        sims = vecs @ vecs[best]
        order = sorted(core, key=lambda i: (i != best, -float(sims[i])))
        sources = [
            {"name": members[i]["name"], "title": members[i]["title"], "url": members[i]["url"],
             "published_at": published[i], "similarity": float(sims[i]), "image_url": members[i]["image_url"],
             "paywall": bool(members[i]["paywall"])}
            for i in order
        ]
        image = next((_web_url(s["image_url"]) for s in sources if _web_url(s["image_url"])), None)
        last = datetime.fromisoformat(c["last_published_at"])
        story = Story(
            cluster_id=c["id"], headline=sources[0]["title"], url=_web_url(sources[0]["url"]),
            source=sources[0]["name"], source_paywall=sources[0]["paywall"], image_url=image, updated_at=last,
            score=0.0, sources=sources, centroid=centroid(vecs[core]),
            entities=entity_stems([members[i]["title"] for i in core]),
        )
        story.score = score(story.outlets, last, now)
        stories.append(story)
    return stories


def ranked(stories: list[Story]) -> list[Story]:
    """Stories told by more than one outlet, best first by outlets with a recency boost."""
    return sorted((s for s in stories if s.outlets > 1), key=lambda s: s.score, reverse=True)


def top_stories(stories: list[Story], exclude=None) -> list[Story]:
    """The front page: the most covered stories, skipping those `exclude(story)` rejects."""
    front = []
    for story in ranked(stories):
        if exclude and exclude(story):
            continue
        front.append(story)
        if len(front) == MAX_STORIES:
            break
    return front


def foreign_filter(others: list[Story]):
    """A predicate that is true for stories about the rest of the world, given the other editions' stories."""
    candidates = [(s.centroid, s.entities) for s in others if s.centroid is not None]
    return lambda s: judge([x["url"] for x in s.sources], s.centroid, s.entities, candidates).foreign


def _nav(countries: list[Country], current: Country, root: str) -> list[dict]:
    """Tab links relative to the current page, so the site works from any folder."""
    return [
        {"name": c.name, "active": c.code == current.code,
         "href": root + ("index.html" if c is countries[0] else f"{c.code}/index.html")}
        for c in countries
    ]


def render_page(
    stories: list[Story],
    now: datetime,
    country: Country = DEFAULT_COUNTRY,
    countries: list[Country] | None = None,
    root: str = "",
) -> str:
    countries = countries or [country]
    return _env.get_template("index.html").render(
        stories=top_stories(stories), now=now, country=country, root=root,
        nav=_nav(countries, country, root), L=LABELS[country.lang],
        fecha=(fecha_en if country.lang == "en" else fecha_es)(now, country.tz),
        hora=lambda dt: hora_local(dt, country.tz),
    )


def _write_atomic(target: Path, html: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, target)  # atomic: readers never see a half-written page


def write_site(
    conn: sqlite3.Connection, model: str, out_dir: Path, countries: list[Country] | None = None
) -> list[Path]:
    """One page per country: the first is site/index.html, the rest site/<code>/index.html."""
    countries = countries or [DEFAULT_COUNTRY]
    now = datetime.now(UTC)
    paths = []
    built = {c.code: build_stories(conn, model, now, c.code) for c in countries}
    for i, country in enumerate(countries):
        exclude = None
        if country.exclude_foreign:  # world news belongs to the other editions
            others = [s for c in countries if c.code != country.code for s in ranked(built[c.code])[:OTHERS_TOP]]
            exclude = foreign_filter(others)
        stories = top_stories(built[country.code], exclude)
        home = i == 0
        html = render_page(stories, now, country, countries, root="" if home else "../")
        target = out_dir / "index.html" if home else out_dir / country.code / "index.html"
        _write_atomic(target, html)
        paths.append(target)
    return paths
