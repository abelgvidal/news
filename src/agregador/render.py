"""Render the front page as static HTML."""
import json
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

ACTIVE_HOURS = 48
HALF_LIFE_HOURS = 24
MAX_STORIES = 10

DEFAULT_COUNTRY = Country("es", "España", "Europe/Madrid")
_DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"]

_env = Environment(loader=PackageLoader("agregador"), autoescape=select_autoescape())


def fecha_es(dt: datetime, tz: str) -> str:
    d = dt.astimezone(ZoneInfo(tz))
    return f"{_DAYS[d.weekday()]}, {d.day} de {_MONTHS[d.month - 1]} de {d.year}"


def hora_local(dt: datetime, tz: str) -> str:
    return dt.astimezone(ZoneInfo(tz)).strftime("%H:%M")


@dataclass
class Story:
    cluster_id: int
    headline: str
    url: str
    image_url: str | None
    updated_at: datetime
    score: float
    sources: list[dict]  # name, title, snippet, url, published_at
    summary: str | None = None
    covering: list[str] | None = None  # outlets that tell the main event (set once summarized)
    skip: bool = False  # the texts share no main event: never shown

    @property
    def covered_sources(self) -> list[dict]:
        """The articles that actually tell the main event; all of them until the story is summarized."""
        if self.covering is None:
            return self.sources
        wanted = set(self.covering)
        return [s for s in self.sources if s["name"] in wanted]

    @property
    def outlets(self) -> int:
        return len(self.outlet_names)

    @property
    def outlet_names(self) -> list[str]:
        return sorted({s["name"] for s in self.covered_sources})

    @property
    def outlet_links(self) -> list[dict]:
        """One entry per outlet, most representative first, linking to that outlet's article.

        Feed content is untrusted, so only http(s) URLs become links; an outlet without a
        safe URL is still listed, just not linked.
        """
        links: dict[str, str | None] = {}
        for src in self.covered_sources:  # sources are ordered most representative first
            url = src["url"] if urlsplit(src["url"]).scheme in ("http", "https") else None
            if links.get(src["name"]) is None:
                links[src["name"]] = url
        return [{"name": name, "url": url} for name, url in links.items()]


def score(outlets: int, last_published: datetime, now: datetime) -> float:
    """Distinct outlets, halved for every HALF_LIFE_HOURS since the latest article."""
    age_h = max((now - last_published).total_seconds() / 3600, 0)
    return outlets * math.pow(0.5, age_h / HALF_LIFE_HOURS)


def build_stories(
    conn: sqlite3.Connection, model: str, now: datetime, country: str = "es"
) -> list[Story]:
    since = (now - timedelta(hours=ACTIVE_HOURS)).isoformat()
    clusters = conn.execute(
        "SELECT id, vector_sum, last_published_at FROM cluster "
        "WHERE model = ? AND country = ? AND last_published_at >= ?", (model, country, since)
    ).fetchall()
    stories = []
    for c in clusters:
        members = conn.execute(
            "SELECT a.title, a.summary, a.url, a.image_url, a.published_at, f.name, e.vector "
            "FROM cluster_member m JOIN article a ON a.id = m.article_id "
            "JOIN feed f ON f.id = a.feed_id JOIN embedding e ON e.article_id = a.id "
            "WHERE m.cluster_id = ? ORDER BY a.published_at DESC", (c["id"],)
        ).fetchall()
        if not members:
            continue
        centroid = np.frombuffer(c["vector_sum"], dtype=np.float32)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        vecs = np.stack([np.frombuffer(m["vector"], dtype=np.float32) for m in members])
        sims = vecs @ centroid
        best = members[int(np.argmax(sims))]
        last = datetime.fromisoformat(c["last_published_at"])
        sources = [
            {"name": m["name"], "title": m["title"], "snippet": m["summary"], "url": m["url"],
             "published_at": datetime.fromisoformat(m["published_at"]), "similarity": float(sim)}
            for m, sim in sorted(zip(members, sims), key=lambda t: t[1], reverse=True)
        ]  # most representative first
        outlets = len({m["name"] for m in members})
        stories.append(Story(
            cluster_id=c["id"], headline=best["title"], url=best["url"],
            image_url=best["image_url"] or next((m["image_url"] for m in members if m["image_url"]), None),
            updated_at=last, score=score(outlets, last, now), sources=sources,
        ))
    return stories


def ranked(stories: list[Story]) -> list[Story]:
    """Stories told by more than one outlet, best first by outlets with a recency boost."""
    return sorted((s for s in stories if s.outlets > 1), key=lambda s: s.score, reverse=True)


def top_stories(stories: list[Story]) -> list[Story]:
    """The front page: ranked, without stories whose texts share no main event."""
    return [s for s in ranked(stories) if not s.skip][:MAX_STORIES]


def attach_summaries(conn: sqlite3.Connection, stories: list[Story], now: datetime | None = None) -> None:
    """Add the stored summary to each story and rank by the outlets that really cover it."""
    now = now or datetime.now(UTC)
    for s in stories:
        row = conn.execute(
            "SELECT headline, text, outlets, status FROM summary WHERE cluster_id = ?", (s.cluster_id,)
        ).fetchone()
        if not row:
            continue
        if row["status"] == "skip":
            s.skip = True
            continue
        s.headline, s.summary = row["headline"], row["text"]
        covering = json.loads(row["outlets"] or "[]")
        if covering:  # older rows have none: keep every outlet of the cluster
            s.covering = covering
            s.score = score(s.outlets, s.updated_at, now)


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
        nav=_nav(countries, country, root), fecha=fecha_es(now, country.tz),
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
    for i, country in enumerate(countries):
        stories = build_stories(conn, model, now, country.code)
        attach_summaries(conn, stories, now)
        stories = top_stories(stories)
        home = i == 0
        html = render_page(stories, now, country, countries, root="" if home else "../")
        target = out_dir / "index.html" if home else out_dir / country.code / "index.html"
        _write_atomic(target, html)
        paths.append(target)
    return paths
