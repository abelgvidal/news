"""Measure cluster quality against tests/labeled_clusters.csv, with the same code the page uses.

Self-contained: embeds the labeled articles (title + summary, like production) with the local model, so
it does not depend on the database (articles are purged after 30 days). Usage:

    uv run python scripts/eval_clusters.py [model] [--certain] [--file tests/labeled_clusters_int.csv]

--certain drops the articles marked uncertain; --file picks the labeled set (default: the Spanish one, the
other is the English International edition). The labels are a first pass by a language model: review them.
"""
import collections
import csv
import sys
from datetime import UTC, datetime

import numpy as np

from agregador.embed import DEFAULT_MODEL, article_text, encode
from agregador.representative import choose_story

argv = sys.argv[1:]
path = "tests/labeled_clusters.csv"
if "--file" in argv:
    i = argv.index("--file"); path = argv[i + 1]; del argv[i:i + 2]
args = [a for a in argv if not a.startswith("--")]
model = args[0] if args else DEFAULT_MODEL
rows = list(csv.DictReader(open(path)))
if "--certain" in sys.argv:
    rows = [r for r in rows if r["uncertain"] != "1"]
vecs = dict(zip((r["article_id"] for r in rows), encode([article_text(r["title"], r["summary"]) for r in rows], model)))
by = collections.defaultdict(list)
for r in rows:
    by[r["cluster_id"]].append(r)

tp = fp = fn = 0
err_all, err_core, off2_all, off2_core, rep_in_story, rep_in_story_plain = [], [], 0, 0, 0, 0
for cid, v in by.items():
    X = np.stack([vecs[r["article_id"]] for r in v])
    titles = [r["title"] for r in v]
    published = [datetime(2026, 1, 1, tzinfo=UTC)] * len(v)
    rep, core_list = choose_story(titles, X, published)
    core = set(core_list)
    real = len({r["outlet"] for r in v if r["belongs"] == "1"})
    all_n = len({r["outlet"] for r in v})
    core_n = len({v[i]["outlet"] for i in core})
    err_all.append(abs(all_n - real)); err_core.append(abs(core_n - real))
    off2_all += abs(all_n - real) >= 2; off2_core += abs(core_n - real) >= 2
    rep_in_story += v[rep]["belongs"] == "1"
    for i, r in enumerate(v):
        b = r["belongs"] == "1"
        tp += b and i in core; fp += (not b) and i in core; fn += b and i not in core
n = len(by)
purity = sum(r["belongs"] == "1" for r in rows) / len(rows)
print(f"{model} | {len(rows)} articles, {n} clusters | purity of the raw clusters: {purity:.0%}")
print(f"outlet count, mean error per cluster : raw {np.mean(err_all):.2f}  ->  core {np.mean(err_core):.2f}")
print(f"clusters off by 2 or more outlets    : raw {off2_all}  ->  core {off2_core}  (of {n})")
print(f"core membership                      : precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f}")
print(f"representative inside the story      : {rep_in_story} of {n}")
