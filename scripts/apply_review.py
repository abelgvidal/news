"""Merge your decisions from the review file back into tests/labeled_clusters.csv.

    uv run python scripts/apply_review.py data/revisar_dudosos.csv

In the review file, fill the column `tu_decision` (the column "mi etiqueta" is only the model's opinion and is
never read): "si" or "no" for whether the article belongs to the story, or "ok" to confirm the model's label.
Rows left empty are untouched. Decided articles get uncertain=0 and labeler=abel.
"""
import csv
import sys
import unicodedata

path = sys.argv[1] if len(sys.argv) > 1 else "data/revisar_dudosos.csv"
fold = lambda t: unicodedata.normalize("NFKD", t.strip().lower()).encode("ascii", "ignore").decode()
YES, NO, OK = {"si", "s", "1", "yes", "y"}, {"no", "n", "0"}, {"ok", "="}

review = {r["article_id"]: fold(r["tu_decision"]) for r in csv.DictReader(open(path)) if r["tu_decision"].strip()}
rows = list(csv.DictReader(open("tests/labeled_clusters.csv")))
changed = flipped = 0
unknown = [v for v in review.values() if v not in YES | NO | OK]
if unknown:
    sys.exit(f"Valores no válidos en tu_decision: {sorted(set(unknown))}. Usa 'si', 'no' u 'ok'.")
for r in rows:
    d = review.get(r["article_id"])
    if d is None:
        continue
    new = r["belongs"] if d in OK else ("1" if d in YES else "0")   # "ok" keeps the model's label
    flipped += new != r["belongs"]
    r.update(belongs=new, uncertain="0", labeler="abel")
    changed += 1
with open("tests/labeled_clusters.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{changed} etiquetas actualizadas; {flipped} cambiaron respecto a mi primera pasada.")
print("Ahora mide el efecto:  uv run python scripts/eval_clusters.py")
