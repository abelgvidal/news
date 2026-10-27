"""Render the front page as static HTML: each story is a cluster's most representative real headline."""
import math
import os
from collections import Counter
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape

from .config import Country
from .countries import flag, involved_countries, primary_country
from .foreign import centroid, entity_stems, judge
from .representative import choose_story, headline_to_show

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
    "es": {"site": "The Daily Ten", "outlets": "medios", "also_in": "También en", "more": "y {n} más",
           "dateline": "Lo más citado en los medios · actualizado a las", "paywall": "Puede requerir suscripción",
           "empty": "Todavía no hay noticias agrupadas.", "no_country": "Sin país identificado",
           "edition": "Lo más citado en los medios · edición de las", "image": "Imagen", "archive": "Archivo",
           "prev": "Anterior", "next": "Siguiente", "latest": "Última edición",
           "none": "Todavía no hay ediciones.", "cover": "Hoy cubrimos", "and": "y"},
    "en": {"site": "The Daily Ten", "outlets": "outlets", "also_in": "Also in", "more": "and {n} more",
           "dateline": "Most covered in the press · updated at", "paywall": "May require a subscription",
           "empty": "No grouped stories yet.", "no_country": "No country identified",
           "edition": "Most covered in the press · edition of", "image": "Image", "archive": "Archive",
           "prev": "Previous", "next": "Next", "latest": "Latest edition",
           "none": "No editions yet.", "cover": "Today we cover", "and": "and"},
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
    image: dict | None = None  # the saved thumbnail (see images.py and edition.py), None when the story has none
    fixed_involved: list[str] | None = None  # countries frozen when an edition was published

    @property
    def outlets(self) -> int:
        return len({s["name"] for s in self.sources})

    @property
    def involved(self) -> list[str]:
        """The countries the story involves, heaviest first (empty when the headlines name none)."""
        if self.fixed_involved is not None:
            return self.fixed_involved
        return involved_countries([s["title"] for s in self.sources])

    @property
    def flags(self) -> str:
        """The flags of the countries involved, or a globe when the headlines name none."""
        return "".join(flag(c) for c in self.involved) or flag(None)

    @property
    def country(self) -> str | None:
        """The country the story is about (None if unclear or mixed), from its headlines."""
        return primary_country([s["title"] for s in self.sources])

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
    conn: sqlite3.Connection, model: str, now: datetime, country: str = "es", active_hours: int = ACTIVE_HOURS
) -> list[Story]:
    start = now - timedelta(hours=active_hours)
    clusters = conn.execute(
        "SELECT id, last_published_at FROM cluster "
        "WHERE model = ? AND country = ? AND last_published_at >= ?", (model, country, start.isoformat())
    ).fetchall()
    stories = []
    for c in clusters:
        members = conn.execute(
            "SELECT a.title, a.url, a.image_url, a.published_at, f.name, f.paywall, e.vector "
            "FROM cluster_member m JOIN article a ON a.id = m.article_id "
            "JOIN feed f ON f.id = a.feed_id JOIN embedding e ON e.article_id = a.id AND e.model = ? "
            "WHERE m.cluster_id = ?", (model, c["id"])
        ).fetchall()
        members = [m for m in members if datetime.fromisoformat(m["published_at"]) >= start]  # not older coverage
        if not members:
            continue
        vecs = np.stack([np.frombuffer(m["vector"], dtype=np.float32) for m in members])
        published = [datetime.fromisoformat(m["published_at"]) for m in members]
        titles = [m["title"] for m in members]
        rep, core = choose_story(titles, vecs, published)  # the cluster may hold neighbouring topics: keep the story
        best = headline_to_show(titles, core, rep)  # same story and outlet count, a headline that is not an outlier
        sims = vecs @ vecs[best]
        order = sorted(core, key=lambda i: (i != best, -float(sims[i])))
        sources = [
            {"name": members[i]["name"], "title": members[i]["title"], "url": members[i]["url"],
             "published_at": published[i], "similarity": float(sims[i]), "image_url": members[i]["image_url"],
             "paywall": bool(members[i]["paywall"])}
            for i in order
        ]
        image = next((_web_url(s["image_url"]) for s in sources if _web_url(s["image_url"])), None)
        last = max(published)
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


def coverage_line(stories: list[Story], L: dict) -> str:
    """"Israel, Brazil and Iran": the main country of each story, in the stories' order, all of them.

    Only the heaviest country of a story counts (the others are mostly actors, or noise from the
    headlines), so the order is the popularity of the stories. Empty when no story names a country.
    A name keeps its words together (no break inside "United States"); the page puts "Today we cover" before.
    """
    names: list[str] = []
    for story in stories:
        if story.involved and story.involved[0] not in names:
            names.append(story.involved[0])
    names = [n.replace(" ", " ") for n in names]
    if len(names) > 1:
        return f"{', '.join(names[:-1])} {L['and']} {names[-1]}."
    return f"{names[0]}." if names else ""


def top_stories(stories: list[Story], exclude=None, max_per_country: int = 0) -> list[Story]:
    """The front page: the most covered stories, skipping those `exclude(story)` rejects.

    With `max_per_country`, a country already holding that many stories gets no more (the next stories
    take the place); stories about no clear country are never limited.
    """
    front, per_country = [], Counter()
    for story in ranked(stories):
        if exclude and exclude(story):
            continue
        country = story.country if max_per_country else None
        if country and per_country[country] >= max_per_country:
            continue
        if country:
            per_country[country] += 1
        front.append(story)
        if len(front) == MAX_STORIES:
            break
    return front


def foreign_filter(others: list[Story]):
    """A predicate that is true for stories about the rest of the world, given the other editions' stories."""
    candidates = [(s.centroid, s.entities) for s in others if s.centroid is not None]
    return lambda s: judge([x["url"] for x in s.sources], s.centroid, s.entities, candidates).foreign


def front_pages(
    conn: sqlite3.Connection, model: str, now: datetime, countries: list[Country], active_hours: int = ACTIVE_HOURS
) -> dict[str, list[Story]]:
    """The front page stories of every edition (country), keyed by country code."""
    built = {c.code: build_stories(conn, model, now, c.code, active_hours) for c in countries}
    pages = {}
    for country in countries:
        exclude = None
        if country.exclude_foreign:  # world news belongs to the other editions
            others = [s for c in countries if c.code != country.code for s in ranked(built[c.code])[:OTHERS_TOP]]
            exclude = foreign_filter(others)
        pages[country.code] = top_stories(built[country.code], exclude, country.max_per_country)
    return pages


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
    edition: dict | None = None,
) -> str:
    """One page. `edition` ({assets, prev, next, archive}: hrefs from this page) marks a published edition."""
    countries = countries or [country]
    front = top_stories(stories)
    labels = LABELS[country.lang]
    return _env.get_template("index.html").render(
        stories=front, now=now, country=country, root=root, edition=edition,
        coverage=coverage_line(front, labels) if country.show_flags else "",
        assets=edition["assets"] if edition else "",
        nav=_nav(countries, country, root), L=labels,
        fecha=(fecha_en if country.lang == "en" else fecha_es)(now, country.tz),
        hora=lambda dt: hora_local(dt, country.tz),
    )


def _write_atomic(target: Path, html: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, target)  # atomic: readers never see a half-written page


def write_site(
    conn: sqlite3.Connection, model: str, out_dir: Path, countries: list[Country] | None = None,
    now: datetime | None = None,
) -> list[Path]:
    """One page per country: the first is site/index.html, the rest site/<code>/index.html."""
    countries = countries or [DEFAULT_COUNTRY]
    now = now or datetime.now(UTC)
    paths = []
    pages = front_pages(conn, model, now, countries)
    for i, country in enumerate(countries):
        stories = pages[country.code]
        home = i == 0
        html = render_page(stories, now, country, countries, root="" if home else "../")
        target = out_dir / "index.html" if home else out_dir / country.code / "index.html"
        _write_atomic(target, html)
        paths.append(target)
    return paths
