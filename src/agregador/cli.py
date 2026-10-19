import argparse
import logging
from pathlib import Path

from . import db
from .config import load_countries, load_feeds
from .cluster import Params, cluster_pending, reset_clusters
from .embed import DEFAULT_MODEL, embed_pending
from .fetch import fetch_all
from .render import attach_summaries, build_stories, top_stories, write_site
from .summarize import run_stats, summarize_stories

DEFAULT_DB = Path("data/agregador.db")
DEFAULT_FEEDS = Path("feeds.toml")
DEFAULT_SITE = Path("site")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agregador")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="fetch all feeds and store new articles")
    sub.add_parser("health", help="show per-feed health")
    sub.add_parser("embed", help="embed articles that have no vector yet")
    p_render = sub.add_parser("render", help="write the static front page")
    p_render.add_argument("--out", type=Path, default=DEFAULT_SITE)
    p_run = sub.add_parser("run", help="full cycle: fetch, purge, embed, cluster, render")
    p_run.add_argument("--out", type=Path, default=DEFAULT_SITE)
    sub.add_parser("summarize", help="write summaries for the top stories (needs Claude credentials)")
    p_stats = sub.add_parser("stats", help="summary volume, retries, fallbacks and estimated spend")
    p_stats.add_argument("--days", type=int, default=7)
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
        for r in conn.execute(
            "SELECT name, last_success_at, consecutive_failures, last_error FROM feed ORDER BY name"
        ):
            print(f"{r['name']:<20} ok={r['last_success_at']} fails={r['consecutive_failures']} {r['last_error'] or ''}")
    elif args.command == "render":
        for path in write_site(conn, DEFAULT_MODEL, args.out, load_countries(args.feeds)):
            print(path)
    elif args.command == "run":
        new = fetch_all(conn, load_feeds(args.feeds))
        purged = db.purge_old(conn)
        embedded = embed_pending(conn)
        assigned, clusters = cluster_pending(conn, DEFAULT_MODEL)
        countries = load_countries(args.feeds)
        calls = summarize_top(conn, countries)
        pages = write_site(conn, DEFAULT_MODEL, args.out, countries)
        logging.info("run: %d new, %d purged, %d embedded, %d assigned (%d new clusters), "
                     "%d summaries -> %d pages", new, purged, embedded, assigned, clusters,
                     calls, len(pages))
    elif args.command == "summarize":
        print(f"{summarize_top(conn, load_countries(args.feeds))} summaries written")
    elif args.command == "stats":
        st = run_stats(conn, args.days)
        if not st["attempted"]:
            print(f"No summary runs in the last {args.days} days.")
        else:
            print(f"Last {args.days} days: {st['runs']} runs, {st['attempted']} summaries attempted "
                  f"({st['attempted_per_day']:.0f}/day), {st['stored']} stored, "
                  f"{st['unverified']} fell back to the original headline ({st['fallback_rate']:.0%})")
            print(f"Drafts per summary: {st['drafts_per_story']:.2f} | verifications: {st['ver']} | "
                  f"escalated to the stronger writer: {st['esc']}")
            print(f"Estimated spend: ${st['cost']:.3f} total, ${st['cost_per_day']:.3f}/day, "
                  f"~${st['cost_per_month']:.2f}/month")
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


def summarize_top(conn, countries) -> int:
    """Summarize the stories that will be on each front page. Never aborts the run."""
    from datetime import UTC, datetime

    calls = 0
    for country in countries:
        stories = top_stories(build_stories(conn, DEFAULT_MODEL, datetime.now(UTC), country.code))
        try:
            calls += summarize_stories(conn, stories)
        except Exception as exc:  # e.g. no credentials: pages fall back to original headlines
            logging.warning("summaries skipped for %s: %s", country.code, exc)
    return calls


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
