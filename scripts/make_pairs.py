"""Sample cross-outlet article pairs across similarity bands for manual labeling.

Writes tests/labeled_pairs.csv (label column empty: fill with 1 = same story, 0 = different)
and data/pairs_similarity.csv (hidden scores, so labeling isn't biased by them).
"""
import csv
import random
import sqlite3
from pathlib import Path

import numpy as np

random.seed(7)
conn = sqlite3.connect("data/agregador.db")
rows = conn.execute(
    "SELECT a.id, a.feed_id, a.title, a.published_at, e.vector FROM article a "
    "JOIN embedding e ON e.article_id = a.id"
).fetchall()
ids = [r[0] for r in rows]
feeds = np.array([r[1] for r in rows])
vecs = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
sims = vecs @ vecs.T

bands = [(0.92, 1.01, 10), (0.85, 0.92, 20), (0.80, 0.85, 25), (0.75, 0.80, 25),
         (0.70, 0.75, 20), (0.60, 0.70, 15), (0.40, 0.60, 10)]
iu = np.triu_indices(len(ids), k=1)
mask = feeds[iu[0]] != feeds[iu[1]]
pi, pj, ps = iu[0][mask], iu[1][mask], sims[iu][mask]

chosen = []
for lo, hi, n in bands:
    idx = np.where((ps >= lo) & (ps < hi))[0]
    pick = random.sample(list(idx), min(n, len(idx)))
    chosen += [(pi[k], pj[k], ps[k]) for k in pick]
random.shuffle(chosen)

Path("tests").mkdir(exist_ok=True)
with open("tests/labeled_pairs.csv", "w", newline="") as f, open("data/pairs_similarity.csv", "w", newline="") as g:
    w, ws = csv.writer(f), csv.writer(g)
    w.writerow(["article_a", "article_b", "title_a", "title_b", "label"])
    ws.writerow(["article_a", "article_b", "similarity"])
    for i, j, s in chosen:
        w.writerow([ids[i], ids[j], rows[i][2], rows[j][2], ""])
        ws.writerow([ids[i], ids[j], f"{s:.4f}"])
print(len(chosen), "pairs written")
