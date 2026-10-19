"""Sample hard pairs (borderline similarity for each model) not yet in tests/labeled_pairs.csv."""
import csv
import random
import sqlite3

import numpy as np

from agregador.embed import article_text, encode

random.seed(11)
conn = sqlite3.connect("data/agregador.db")
rows = conn.execute("SELECT id, feed_id, title, summary FROM article").fetchall()
existing = {(int(r["article_a"]), int(r["article_b"])) for r in csv.DictReader(open("tests/labeled_pairs.csv"))}
texts = [article_text(r[2], r[3]) for r in rows]
feeds = np.array([r[1] for r in rows])
iu = np.triu_indices(len(rows), k=1)
mask = feeds[iu[0]] != feeds[iu[1]]
pi, pj = iu[0][mask], iu[1][mask]

chosen = {}
for model, lo, hi, n in [("intfloat/multilingual-e5-small", 0.84, 0.92, 40),
                         ("paraphrase-multilingual-MiniLM-L12-v2", 0.65, 0.80, 40)]:
    v = encode(texts, model)
    s = np.einsum("ij,ij->i", v[pi], v[pj])
    idx = [k for k in np.where((s >= lo) & (s < hi))[0] if (rows[pi[k]][0], rows[pj[k]][0]) not in existing]
    for k in random.sample(idx, min(n, len(idx))):
        chosen[(rows[pi[k]][0], rows[pj[k]][0])] = (rows[pi[k]][2], rows[pj[k]][2])
items = list(chosen.items()); random.shuffle(items)
with open("data/hard_pairs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["article_a", "article_b", "title_a", "title_b"])
    for (a, b), (ta, tb) in items: w.writerow([a, b, ta, tb])
print(len(items), "pairs")
