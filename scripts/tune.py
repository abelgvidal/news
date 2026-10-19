"""Pick the best similarity threshold per model from tests/labeled_pairs.csv."""
import csv
import sys

import numpy as np

from agregador.embed import DEFAULT_MODEL, article_text, encode

import sqlite3

conn = sqlite3.connect("data/agregador.db")
pairs = list(csv.DictReader(open("tests/labeled_pairs.csv")))
ids = sorted({int(p[k]) for p in pairs for k in ("article_a", "article_b")})
text = {r[0]: article_text(r[1], r[2]) for r in conn.execute(
    f"SELECT id, title, summary FROM article WHERE id IN ({','.join('?'*len(ids))})", ids)}
if "--certain" in sys.argv:
    sys.argv.remove("--certain")
    pairs = [p for p in pairs if p["uncertain"] != "1"]
models = sys.argv[1:] or [DEFAULT_MODEL]
y = np.array([int(p["label"]) for p in pairs])

for model in models:
    order = list(text)
    vecs = dict(zip(order, encode([text[i] for i in order], model)))
    s = np.array([float(vecs[int(p["article_a"])] @ vecs[int(p["article_b"])]) for p in pairs])
    print(f"\n{model}")
    best = None
    for t in np.arange(0.50, 0.96, 0.01):
        pred = s >= t
        tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # precision matters more than recall: a wrong merge is worse than a missed one
        fb = (1 + 0.5**2) * prec * rec / (0.5**2 * prec + rec) if prec + rec else 0.0
        if best is None or fb > best[0]:
            best = (fb, t, prec, rec, f1)
        if round(t * 100) % 5 == 0:
            print(f"  t={t:.2f}  precision={prec:.2f}  recall={rec:.2f}  f1={f1:.2f}")
    print(f"  best F0.5={best[0]:.3f} at t={best[1]:.2f} (precision={best[2]:.2f}, recall={best[3]:.2f})")
