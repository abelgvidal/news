"""Measure the rule that keeps world news out of the Spain edition (src/agregador/foreign.py).

    uv run python scripts/eval_foreign.py export [--top 30]   # add today's Spanish stories to tests/labeled_foreign.csv
    uv run python scripts/eval_foreign.py score               # precision / recall of the current rule on the labeled rows

`export` stores each story's signals (share of articles in international URL sections, similarity with the
best match in the other editions, shared proper nouns), so the file is self-contained and keeps working after the
database rotates. Fill `is_foreign` (1 = the story is about the rest of the world and should not be in Spain, 0 = it
belongs there) and mark doubtful ones with `uncertain=1`. The first pass of labels is the model's: review them.
"""
import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from agregador import db
from agregador.embed import DEFAULT_MODEL
from agregador.foreign import CROSS_COS, CROSS_ENTITIES, SECTION_SHARE
from agregador.foreign import judge
from agregador.render import OTHERS_TOP, build_stories, ranked

PATH = "tests/labeled_foreign.csv"
FIELDS = ["date", "headline", "outlets", "section_share", "best_cos", "shared", "rule_foreign", "is_foreign", "uncertain", "labeler"]


def export(top: int) -> None:
    conn = db.connect(Path("data/agregador.db"))
    now = datetime.now(UTC)
    others = [(s.centroid, s.entities) for s in ranked(build_stories(conn, DEFAULT_MODEL, now, "int"))[:OTHERS_TOP]]
    rows = list(csv.DictReader(open(PATH))) if os.path.exists(PATH) else []
    seen = {(r["date"], r["headline"]) for r in rows}
    today, added = now.strftime("%Y-%m-%d"), 0
    for s in ranked(build_stories(conn, DEFAULT_MODEL, now, "es"))[:top]:
        if (today, s.headline) in seen:
            continue
        v = judge([x["url"] for x in s.sources], s.centroid, s.entities, others)
        rows.append({"date": today, "headline": s.headline, "outlets": s.outlets, "section_share": f"{v.section_share:.3f}",
                     "best_cos": f"{v.best_cos:.3f}", "shared": v.shared, "rule_foreign": int(v.foreign),
                     "is_foreign": "", "uncertain": "", "labeler": ""})
        added += 1
    with open(PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"{added} stories added to {PATH} ({len(rows)} rows, {sum(1 for r in rows if r['is_foreign'] == '')} still unlabeled)")


def score() -> None:
    rows = [r for r in csv.DictReader(open(PATH)) if r["is_foreign"] in ("0", "1")]
    if not rows:
        sys.exit("No labeled rows yet.")

    def predict(r):  # recomputed from the stored signals with the thresholds in foreign.py
        return float(r["section_share"]) >= SECTION_SHARE or (float(r["best_cos"]) >= CROSS_COS and int(r["shared"]) >= CROSS_ENTITIES)

    for label, subset in (("all rows", rows), ("without uncertain", [r for r in rows if r["uncertain"] != "1"])):
        tp = sum(predict(r) and r["is_foreign"] == "1" for r in subset)
        fp = sum(predict(r) and r["is_foreign"] == "0" for r in subset)
        fn = sum((not predict(r)) and r["is_foreign"] == "1" for r in subset)
        print(f"{label}: {len(subset)} stories, {sum(r['is_foreign'] == '1' for r in subset)} foreign | "
              f"precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f} (tp {tp}, fp {fp}, fn {fn})")
    for r in rows:
        if predict(r) != (r["is_foreign"] == "1"):
            print(f"  {'MISSED ' if r['is_foreign'] == '1' else 'WRONGLY EXCLUDED '}{r['date']} sec {r['section_share']} cos {r['best_cos']} sh {r['shared']} | {r['headline'][:80]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export(int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30)
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)
