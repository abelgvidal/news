"""Pick the article that stands for a cluster, with deterministic rules (no generative AI).

The representative is the article most similar to all the others (the medoid, not the one nearest
the average), minus a penalty for headlines that make poor front-page text: quotes, questions, live
blogs, all caps, opinion labels, very long ones. Ties go to the article published first.
"""
import re
from datetime import datetime

import numpy as np

# Centrality scores sit in a narrow band (embedding similarities of the same story are about
# 0.85-0.95), so one penalty has to weigh about as much as the whole spread of that band.
PENALTY_WEIGHT = 0.03
MAX_HEADLINE_CHARS = 110

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


def pick_representative(titles: list[str], vectors: np.ndarray, published: list[datetime]) -> int:
    """Index of the article that should represent the cluster."""
    penalties = np.array([headline_penalties(t) for t in titles])
    scores = centrality(vectors) - PENALTY_WEIGHT * penalties
    return max(range(len(titles)), key=lambda i: (round(float(scores[i]), 6), -published[i].timestamp()))
