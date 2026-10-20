"""Measure which country the dictionary (src/agregador/countries.toml) assigns to International stories.

    uv run python scripts/eval_countries.py export [--top 30]   # add today's International stories
    uv run python scripts/eval_countries.py score               # accuracy of the current dictionary

`export` stores the headlines of each story's articles, so the file is self-contained and the score is recomputed
with the dictionary as it is now. Fill `label` with the country the story is about, spelled as in countries.toml
(the country where it happens or that it is mainly about), or leave it empty when there is no single country (a
summit between three leaders, a general story). Mark doubtful ones with `uncertain=1`. First pass by a language model: review.
"""
import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from agregador import db
from agregador.countries import primary_country
from agregador.embed import DEFAULT_MODEL
from agregador.render import build_stories, ranked

PATH = "tests/labeled_countries.csv"
FIELDS = ["date", "headline", "titles", "label", "unlabeled", "uncertain", "labeler"]
SEP = " || "


def export(top: int) -> None:
    conn = db.connect(Path("data/agregador.db"))
    now = datetime.now(UTC)
    rows = list(csv.DictReader(open(PATH))) if os.path.exists(PATH) else []
    seen = {(r["date"], r["headline"]) for r in rows}
    today, added = now.strftime("%Y-%m-%d"), 0
    for s in ranked(build_stories(conn, DEFAULT_MODEL, now, "int"))[:top]:
        if (today, s.headline) not in seen:
            rows.append({"date": today, "headline": s.headline, "titles": SEP.join(x["title"] for x in s.sources),
                         "label": "", "unlabeled": 1, "uncertain": "", "labeler": ""})
            added += 1
    with open(PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"{added} stories added to {PATH} ({len(rows)} rows)")


def score() -> None:
    rows = [r for r in csv.DictReader(open(PATH)) if r["unlabeled"] != "1"]
    if not rows:
        sys.exit("No labeled rows yet.")
    for name, subset in (("all rows", rows), ("without uncertain", [r for r in rows if r["uncertain"] != "1"])):
        pred = [(r, primary_country(r["titles"].split(SEP))) for r in subset]
        assigned = [(r, p) for r, p in pred if p]
        right = sum(p == r["label"] for r, p in assigned)
        with_label = [(r, p) for r, p in pred if r["label"]]
        found = sum(p == r["label"] for r, p in with_label)
        print(f"{name}: {len(subset)} stories | assigned {len(assigned)} (right {right}, precision {right / max(len(assigned), 1):.2f}) | "
              f"with a labeled country {len(with_label)} (found {found}, recall {found / max(len(with_label), 1):.2f})")
    for r in rows:
        p = primary_country(r["titles"].split(SEP))
        if p != (r["label"] or None):
            kind = "WRONG" if p and r["label"] else ("INVENTED" if p else "MISSED")
            print(f"  {kind:<8} label {r['label'] or '-':<15} predicted {p or '-':<15} | {r['headline'][:70]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export(int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30)
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)
