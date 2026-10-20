"""Tell whether a story is really about the rest of the world, with deterministic rules (no generative AI).

Used to keep world news out of the Spain edition. Two independent signals:

1. The URL section: Spanish outlets file articles under /internacional/, /mundo/... When enough of a
   story's articles sit in such sections, the story is international.
2. The story already appears in another edition: its centroid is similar to a story of that edition
   (similarity across languages) *and* they share proper nouns (Trump, Putin, París/Paris...). Neither
   alone is enough: across languages the similarity of true and false matches overlaps, and
   a single shared name is common.

Thresholds were set looking at one day of Spanish front pages (tests/labeled_foreign.csv accumulates
labeled days: scripts/eval_foreign.py measures them).
"""
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

INTL_SECTIONS = frozenset({"internacional", "mundo", "international", "world", "europa", "exterior"})
SECTION_SHARE = 0.30  # share of a story's articles filed under an international section
CROSS_COS = 0.91  # similarity with a story of another edition...
CROSS_ENTITIES = 2  # ...and at least this many proper-noun stems in common
ENTITY_BONUS = 0.05  # when picking the best match, each shared stem is worth this much similarity
_STOP = frozenset({"this", "that", "with", "from", "after", "over", "para", "como", "tras", "sobre", "entre",
                   "desde", "hasta", "ante", "contra"})
_CAPITALIZED = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-ZÁÉÍÓÚÑ][\wáéíóúñü]{3,})")


def section(url: str) -> str:
    """First path segment of the URL, lowercase ('' when there is none)."""
    parts = [p for p in urlsplit(url).path.split("/") if p]
    return parts[0].lower() if parts else ""


def section_share(urls: list[str]) -> float:
    """Share of the articles filed under an international section."""
    return sum(section(u) in INTL_SECTIONS for u in urls) / len(urls) if urls else 0.0


def entity_stems(titles: list[str]) -> frozenset[str]:
    """Five-letter stems of the capitalized words after the first one of each headline.

    The stem makes Zelensky/Zelenski, Francia/France or Milán/Milan comparable across Spanish and English.
    """
    out = set()
    for title in titles:
        for m in _CAPITALIZED.finditer(title):
            word = unicodedata.normalize("NFKD", m.group(1).lower()).encode("ascii", "ignore").decode()
            if word not in _STOP:
                out.add(word[:5])
    return frozenset(out)


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Normalized mean of L2-normalized vectors."""
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


@dataclass(frozen=True)
class Verdict:
    foreign: bool
    reason: str  # "section", "match", "section+match" or ""
    section_share: float
    best_cos: float  # similarity with the best match in the other editions (0 if there is none)
    shared: int  # proper-noun stems shared with that match


def judge(
    urls: list[str],
    vector: np.ndarray | None,
    entities: frozenset[str],
    others: list[tuple[np.ndarray, frozenset[str]]],
) -> Verdict:
    """Is this story about the rest of the world? `others` are (centroid, entities) of the other editions' stories."""
    share = section_share(urls)
    best_cos, shared = 0.0, 0
    if vector is not None and others:
        best = max(others, key=lambda o: float(vector @ o[0]) + ENTITY_BONUS * len(entities & o[1]))
        best_cos, shared = float(vector @ best[0]), len(entities & best[1])
    by_section = share >= SECTION_SHARE
    by_match = best_cos >= CROSS_COS and shared >= CROSS_ENTITIES
    reason = "+".join(r for r, hit in (("section", by_section), ("match", by_match)) if hit)
    return Verdict(by_section or by_match, reason, share, best_cos, shared)
