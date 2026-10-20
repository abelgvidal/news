"""Pick the article that stands for a cluster, and which articles really belong to it (no generative AI).

A cluster can hold neighbouring topics. So the story is found in two steps: the medoid (the article most
similar to all the others, not the one nearest the average) fixes the story; its "core" is the articles
similar enough to it in meaning (embedding) and in words (headline overlap). The representative is the most
central article of the core, with a penalty for headlines that make poor front-page text (quotes, questions,
live blogs, all caps, opinion labels, very long ones) that only separates articles almost as central as the
best: applied to everything it pushed the choice to tangential articles, or out of the story in mixed
clusters. Ties go to the article published first. Counts and links use the core.
"""
import re
import unicodedata
from datetime import datetime

import numpy as np

# Centrality scores sit in a narrow band (embedding similarities of the same story are about
# 0.85-0.95), so one penalty has to weigh about as much as the whole spread of that band.
PENALTY_WEIGHT = 0.03
NEAR_CENTER = 0.02  # penalties only separate articles within this much centrality of the most central one
MAX_HEADLINE_CHARS = 110

# Membership of the core: similarity to the representative + WORD_WEIGHT * headline word overlap >= MEMBER_THRESHOLD.
# Tuned on tests/labeled_clusters.csv (scripts/eval_clusters.py): it cut the mean error of the outlet count
# per cluster from 1.20 to 0.47. Re-tune when the embedding model changes.
MEMBER_THRESHOLD = 0.90
WORD_WEIGHT = 0.6
_STOP = set("para como pero sobre entre desde hasta tras ante este esta estos estas esos esas aqui alli donde cuando "
            "porque segun mas menos muy tambien todo toda todos todas otro otra otros otras cada tres dos cinco "
            "cuatro mil millones anos dia dias hoy ayer mismo misma parte tiene tienen puede pueden hace hacen "
            "sera seran fue fueron esta estan estar ser haber hay sido nuevo nueva primero primera ultimo ultima "
            "gran grande".split())

_QUOTES = '"“”«»'
_LIVE = re.compile(r"\b(en directo|en vivo|[uú]ltima hora|minuto a minuto|directo)\b", re.IGNORECASE)
_OPINION = re.compile(r"^(opini[oó]n|editorial|an[aá]lisis|columna|cr[oó]nica)\b", re.IGNORECASE)


def headline_penalties(title: str) -> int:
    """How many reasons make this headline a poor choice for the front page."""
    n = 0
    if any(q in title for q in _QUOTES):
        n += 1
    if "?" in title or "¿" in title:
        n += 1
    if _LIVE.search(title):
        n += 1
    if "|" in title:  # live-blog and section separators
        n += 1
    if len(title) > MAX_HEADLINE_CHARS:
        n += 1
    letters = [c for c in title if c.isalpha()]
    if len(letters) > 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        n += 1
    if _OPINION.match(title):
        n += 1
    return n


def centrality(vectors: np.ndarray) -> np.ndarray:
    """Mean cosine similarity of each (L2-normalized) vector to all the others."""
    n = len(vectors)
    if n == 1:
        return np.ones(1)
    sims = vectors @ vectors.T
    return (sims.sum(axis=1) - np.diag(sims)) / (n - 1)


def title_words(title: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", title.lower()).encode("ascii", "ignore").decode()
    return {w for w in re.findall(r"[a-z0-9]{4,}", folded) if w not in _STOP}


def title_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the content words of two headlines (0 to 1)."""
    wa, wb = title_words(a), title_words(b)
    return len(wa & wb) / max(len(wa | wb), 1)


def core_members(titles: list[str], vectors: np.ndarray, rep: int) -> list[int]:
    """Indices of the articles that belong to the representative's story (always including it)."""
    sims = vectors @ vectors[rep]
    return [
        i for i in range(len(titles))
        if i == rep or sims[i] + WORD_WEIGHT * title_overlap(titles[i], titles[rep]) >= MEMBER_THRESHOLD
    ]


def choose_story(titles: list[str], vectors: np.ndarray, published: list[datetime]) -> tuple[int, list[int]]:
    """The representative article and the indices of the core of its story."""
    medoid = int(np.argmax(centrality(vectors)))
    candidates = core_members(titles, vectors, medoid)
    if len(candidates) > 1:
        local = centrality(vectors[candidates])
        penalties = np.array([headline_penalties(titles[i]) for i in candidates])
        scores = np.where(local >= local.max() - NEAR_CENTER, local - PENALTY_WEIGHT * penalties, -np.inf)
        best = max(range(len(candidates)),
                   key=lambda j: (round(float(scores[j]), 6), -published[candidates[j]].timestamp()))
        rep = candidates[best]
    else:
        rep = medoid
    return rep, core_members(titles, vectors, rep)
