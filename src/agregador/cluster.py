"""Incremental threshold clustering of article embeddings."""
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

_ENTITY_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ][\wáéíóúñü]{3,}|\d{2,})\b")
# Capitalized words that are common sentence starters / generic, not entities.
_STOP_ENTITIES = {
    "este", "esta", "estos", "estas", "para", "como", "cuando", "donde", "tras",
    "ante", "sobre", "según", "desde", "hasta", "entre", "contra", "pero",
    "españa", "gobierno",
}


@dataclass(frozen=True)
class Params:
    threshold: float = 0.90          # min cosine similarity to join a cluster
    same_feed_threshold: float = 0.95  # stricter if the outlet is already in the cluster
    entity_free_threshold: float = 0.94  # above this, no entity overlap needed
    window_hours: int = 48


@dataclass
class Cluster:
    id: int
    vector_sum: np.ndarray
    size: int
    entities: set[str]
    feed_ids: set[int]
    last_published_at: datetime
    country: str = "es"

    @property
    def centroid(self) -> np.ndarray:
        n = np.linalg.norm(self.vector_sum)
        return self.vector_sum / n if n else self.vector_sum


@dataclass
class Assignment:
    cluster_id: int | None
    similarity: float | None
    runner_up_id: int | None
    runner_up_similarity: float | None
    note: str = ""


def extract_entities(title: str, summary: str) -> set[str]:
    """Capitalized words and numbers, skipping the first word of the title."""
    first_stripped = title.split(" ", 1)[1] if " " in title else ""
    found = _ENTITY_RE.findall(f"{first_stripped} {summary}")
    return {e.lower() for e in found if e.lower() not in _STOP_ENTITIES}


def choose(
    vec: np.ndarray,
    entities: set[str],
    feed_id: int,
    candidates: list[Cluster],
    params: Params,
) -> Assignment:
    """Pick the best eligible cluster for an article, or None to start a new one."""
    scored = sorted(
        ((float(vec @ c.centroid), c) for c in candidates), key=lambda t: t[0], reverse=True
    )
    best_id = scored[0][1].id if scored else None
    best_sim = scored[0][0] if scored else None
    for i, (sim, c) in enumerate(scored):
        if sim < params.threshold:
            break
        if feed_id in c.feed_ids and sim < params.same_feed_threshold:
            continue
        if sim < params.entity_free_threshold and entities and c.entities and not (entities & c.entities):
            continue
        runner = scored[i + 1] if i + 1 < len(scored) else None
        return Assignment(
            c.id, sim, runner[1].id if runner else None, runner[0] if runner else None
        )
    note = ""
    if best_sim is not None and best_sim >= params.threshold:
        note = "best candidates rejected by guards"
    return Assignment(None, best_sim, best_id, None, note)


def _load_active(conn: sqlite3.Connection, model: str, since: datetime) -> dict[int, Cluster]:
    rows = conn.execute(
        "SELECT * FROM cluster WHERE model = ? AND last_published_at >= ?",
        (model, since.isoformat()),
    ).fetchall()
    return {
        r["id"]: Cluster(
            id=r["id"],
            vector_sum=np.frombuffer(r["vector_sum"], dtype=np.float32).copy(),
            size=r["size"],
            entities=set(json.loads(r["entities"])),
            feed_ids=set(json.loads(r["feed_ids"])),
            last_published_at=datetime.fromisoformat(r["last_published_at"]),
            country=r["country"],
        )
        for r in rows
    }


def cluster_pending(
    conn: sqlite3.Connection, model: str, params: Params = Params()
) -> tuple[int, int]:
    """Assign every embedded, unclustered article. Returns (assigned, new_clusters).

    Articles only ever join clusters of their own country.
    """
    rows = conn.execute(
        "SELECT a.id, a.feed_id, f.country, a.title, a.summary, a.published_at, e.vector "
        "FROM article a JOIN feed f ON f.id = a.feed_id "
        "JOIN embedding e ON e.article_id = a.id AND e.model = ? "
        "LEFT JOIN cluster_member m ON m.article_id = a.id "
        "WHERE m.article_id IS NULL ORDER BY a.published_at",
        (model,),
    ).fetchall()
    if not rows:
        return 0, 0

    oldest = datetime.fromisoformat(rows[0]["published_at"]) - timedelta(hours=params.window_hours)
    active = _load_active(conn, model, oldest)
    new_clusters = 0

    for r in rows:
        published = datetime.fromisoformat(r["published_at"])
        vec = np.frombuffer(r["vector"], dtype=np.float32)
        entities = extract_entities(r["title"], r["summary"])
        window_start = published - timedelta(hours=params.window_hours)
        candidates = [c for c in active.values()
                      if c.country == r["country"] and c.last_published_at >= window_start]
        result = choose(vec, entities, r["feed_id"], candidates, params)

        if result.cluster_id is None:
            cur = conn.execute(
                "INSERT INTO cluster (model, vector_sum, size, entities, feed_ids, created_at, "
                "last_published_at, country) VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
                (model, vec.tobytes(), json.dumps(sorted(entities)),
                 json.dumps([r["feed_id"]]), published.isoformat(), published.isoformat(),
                 r["country"]),
            )
            cluster = Cluster(cur.lastrowid, vec.copy(), 1, set(entities), {r["feed_id"]},
                              published, r["country"])
            active[cluster.id] = cluster
            new_clusters += 1
        else:
            cluster = active[result.cluster_id]
            cluster.vector_sum = cluster.vector_sum + vec
            cluster.size += 1
            cluster.entities |= entities
            cluster.feed_ids.add(r["feed_id"])
            cluster.last_published_at = max(cluster.last_published_at, published)
            conn.execute(
                "UPDATE cluster SET vector_sum = ?, size = ?, entities = ?, feed_ids = ?, "
                "last_published_at = ? WHERE id = ?",
                (cluster.vector_sum.tobytes(), cluster.size, json.dumps(sorted(cluster.entities)),
                 json.dumps(sorted(cluster.feed_ids)), cluster.last_published_at.isoformat(),
                 cluster.id),
            )
        conn.execute(
            "INSERT INTO cluster_member (article_id, cluster_id, similarity, "
            "runner_up_cluster_id, runner_up_similarity, note) VALUES (?, ?, ?, ?, ?, ?)",
            (r["id"], cluster.id, result.similarity, result.runner_up_id,
             result.runner_up_similarity, result.note),
        )
    conn.commit()
    return len(rows), new_clusters


def reset_clusters(conn: sqlite3.Connection) -> None:
    """Drop all clusters so every article is re-assigned (after a model or parameter change)."""
    conn.execute("DELETE FROM cluster_member")
    conn.execute("DELETE FROM cluster")
    conn.commit()
