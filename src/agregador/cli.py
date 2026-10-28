import argparse
import logging
from pathlib import Path

from . import db
from .config import FeedConfig, load_countries, load_feeds
from .cluster import Params, cluster_pending, reset_clusters
from .edition import make_edition, refresh_site
from .embed import DEFAULT_MODEL, embed_pending
from .fetch import fetch_all
from .render import write_site

DEFAULT_DB = Path("data/agregador.db")
DEFAULT_FEEDS = Path("feeds.toml")
DEFAULT_SITE = Path("site")
DEFAULT_TZ = "Europe/Madrid"  # the day an edition belongs to


def main() -> None:
    parser = argparse.ArgumentParser(prog="agregador")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="fetch all feeds and store new articles")
    sub.add_parser("health", help="show per-feed health")
    sub.add_parser("embed", help="embed articles that have no vector yet")
    p_render = sub.add_parser("render", help="write a live preview of the front pages (no images, not archived)")
    p_render.add_argument("--out", type=Path, default=DEFAULT_SITE)
    sub.add_parser("run", help="collect: fetch, purge, embed, cluster (publishing is `edition`)")
    p_edition = sub.add_parser("edition", help="publish today's edition: pages, images and manifest, then the home page")
    p_edition.add_argument("--out", type=Path, default=DEFAULT_SITE)
    p_edition.add_argument("--tz", default=DEFAULT_TZ)
    p_edition.add_argument("--force", action="store_true", help="replace today's edition if it exists")
    p_edition.add_argument("--no-images", action="store_true")
    p_site = sub.add_parser("site", help="rebuild all pages from the published editions")
    p_site.add_argument("--out", type=Path, default=DEFAULT_SITE)
    p_cluster = sub.add_parser("cluster", help="assign unclustered articles to clusters")
    p_cluster.add_argument("--threshold", type=float, default=Params().threshold)
    p_re = sub.add_parser("recluster", help="drop all clusters and rebuild (re-embeds if the model changed)")
    p_re.add_argument("--threshold", type=float, default=Params().threshold)
    p_show = sub.add_parser("clusters", help="print the largest clusters")
    p_show.add_argument("--top", type=int, default=15)
    p_explain = sub.add_parser("explain", help="explain an article's cluster assignment")
    p_explain.add_argument("article_id", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    conn = db.connect(args.db)
    db.migrate(conn)

    if args.command == "fetch":
        new = fetch_all(conn, load_feeds(args.feeds))
        print(f"{new} new articles")
    elif args.command == "health":
        rows, retired = feed_health(conn, load_feeds(args.feeds))
        for r in rows:
            print(f"{r['name']:<20} ok={r['last_success_at']} fails={r['consecutive_failures']} {r['last_error'] or ''}")
        if retired:
            print(f"({retired} feed rows no longer in {args.feeds} ignored)")
    elif args.command == "render":
        for path in write_site(conn, DEFAULT_MODEL, args.out, load_countries(args.feeds)):
            print(path)
    elif args.command == "run":
        new = fetch_all(conn, load_feeds(args.feeds))
        purged = db.purge_old(conn)
        embedded = embed_pending(conn)
        assigned, clusters = cluster_pending(conn, DEFAULT_MODEL)
        logging.info("run: %d new, %d purged, %d embedded, %d assigned (%d new clusters)",
                     new, purged, embedded, assigned, clusters)
    elif args.command == "edition":
        target = make_edition(conn, DEFAULT_MODEL, load_countries(args.feeds), args.out, args.tz,
                              force=args.force, images=not args.no_images)
        print(target or "today's edition already exists (use --force to replace it)")
    elif args.command == "site":
        refresh_site(args.out)
    elif args.command == "embed":
        print(f"{embed_pending(conn)} articles embedded")
    elif args.command == "cluster":
        assigned, new = cluster_pending(conn, DEFAULT_MODEL, Params(threshold=args.threshold))
        print(f"{assigned} articles assigned, {new} new clusters")
    elif args.command == "recluster":
        reset_clusters(conn)
        print(f"{embed_pending(conn)} articles embedded")
        assigned, new = cluster_pending(conn, DEFAULT_MODEL, Params(threshold=args.threshold))
        print(f"{assigned} articles assigned, {new} new clusters")
    elif args.command == "clusters":
        show_clusters(conn, args.top)
    elif args.command == "explain":
        explain(conn, args.article_id)


def feed_health(conn, feeds: list[FeedConfig]) -> tuple[list, int]:
    """Health rows of the feeds in the config, and how many rows of retired feeds were left out.

    A feed dropped from feeds.toml keeps its row (and its failure count) in the database; showing it would
    be a false alarm, since nothing fetches it any more.
    """
    urls = {f.url for f in feeds}
    rows = conn.execute(
        "SELECT name, url, last_success_at, consecutive_failures, last_error FROM feed ORDER BY name"
    ).fetchall()
    current = [r for r in rows if r["url"] in urls]
    return current, len(rows) - len(current)


def show_clusters(conn, top: int) -> None:
    for c in conn.execute(
        "SELECT c.id, c.size, COUNT(DISTINCT a.feed_id) AS outlets FROM cluster c "
        "JOIN cluster_member m ON m.cluster_id = c.id JOIN article a ON a.id = m.article_id "
        "GROUP BY c.id ORDER BY outlets DESC, c.size DESC LIMIT ?",
        (top,),
    ):
        print(f"\n== cluster {c['id']}: {c['size']} articles, {c['outlets']} outlets")
        for a in conn.execute(
            "SELECT f.name, a.title FROM cluster_member m JOIN article a ON a.id = m.article_id "
            "JOIN feed f ON f.id = a.feed_id WHERE m.cluster_id = ? ORDER BY a.published_at",
            (c["id"],),
        ):
            print(f"  [{a['name']}] {a['title']}")


def explain(conn, article_id: int) -> None:
    r = conn.execute(
        "SELECT a.title, m.cluster_id, m.similarity, m.runner_up_cluster_id, "
        "m.runner_up_similarity, m.note FROM article a "
        "JOIN cluster_member m ON m.article_id = a.id WHERE a.id = ?",
        (article_id,),
    ).fetchone()
    if not r:
        print("article not clustered")
        return
    print(f"{r['title']}\n  cluster {r['cluster_id']}  similarity={r['similarity']}")
    print(f"  runner-up cluster {r['runner_up_cluster_id']}  similarity={r['runner_up_similarity']}")
    if r["note"]:
        print(f"  note: {r['note']}")
