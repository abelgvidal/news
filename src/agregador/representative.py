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
import statistics
import unicodedata
from collections import Counter
from datetime import datetime

import numpy as np

# Centrality scores sit in a narrow band (embedding similarities of the same story are about
# 0.85-0.95), so one penalty has to weigh about as much as the whole spread of that band.
PENALTY_WEIGHT = 0.03
NEAR_CENTER = 0.02  # penalties only separate articles within this much centrality of the most central one
MAX_HEADLINE_CHARS = 110

# The headline shown is the representative's, unless it is unusually long for its own story: then a shorter headline of
# the same core that keeps all the key terms takes its place (the core, the outlet count and the ranking do not change).
# Checked on the six editions published 2026-10-20..25: 1.35 swaps 2 of 120 stories; 1.25 also cut a death toll.
LONG_HEADLINE_RATIO = 1.35  # against the median length of the core's headlines
MIN_CORE_FOR_SWAP = 4  # a median of two or three headlines says nothing about what is long
KEY_TERM_SHARE = 0.4  # a key term appears in at least this share of the core's headlines (and in two or more)

# Membership of the core: similarity to the representative + WORD_WEIGHT * headline word overlap >= MEMBER_THRESHOLD.
# Tuned on tests/labeled_clusters.csv, labeled with a strict "same specific news item" criterion (scripts/eval_clusters.py):
# mean error of the outlet count per cluster 1.60 raw -> 0.67 (leave-one-cluster-out estimate 0.67). Re-tune when the
# embedding model changes or the labeled set grows.
MEMBER_THRESHOLD = 0.92
WORD_WEIGHT = 1.5
_STOP = set("para como pero sobre entre desde hasta tras ante este esta estos estas esos esas aqui alli donde cuando "
            "porque segun mas menos muy tambien todo toda todos todas otro otra otros otras cada tres dos cinco "
            "cuatro mil millones anos dia dias hoy ayer mismo misma parte tiene tienen puede pueden hace hacen "
            "sera seran fue fueron esta estan estar ser haber hay sido nuevo nueva primero primera ultimo ultima "
            "gran grande with from that this have been will after over their about says said what which would could "
            "into than they them were more most when where while also amid just only some other against before "
            "between during under news live updates".split())

_QUOTES = '"“”«»'
_LIVE = re.compile(
    r"\b(en directo|en vivo|[uú]ltima hora|minuto a minuto|directo|live updates|live blog|live|as it happened|"
    r"firstft|podcast|newsletter)\b",
    re.IGNORECASE,
)
_OPINION = re.compile(
    r"^(opini[oó]n|editorial|an[aá]lisis|columna|cr[oó]nica|opinion|analysis|letters|column)\b", re.IGNORECASE
)


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
    unpossessed = re.sub(r"['’]s\b", "", title.lower())  # "Brazil’s" must match "Brazil", not become "brazils"
    folded = unicodedata.normalize("NFKD", unpossessed).encode("ascii", "ignore").decode()
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


def _terms(title: str) -> set[str]:
    """Content words plus every number: a death toll or an amount is part of what a headline says."""
    folded = unicodedata.normalize("NFKD", title.lower()).encode("ascii", "ignore").decode()
    return title_words(title) | set(re.findall(r"\d+", folded))


def headline_to_show(titles: list[str], core: list[int], rep: int) -> int:
    """The article whose headline is shown: `rep`, unless that headline is far longer than the story's usual one.

    Then it is the headline of the core, closest to the median length, that has no front-page penalties, is
    shorter, and contains every key term (words and numbers most of the core's headlines share) the long one does.
    """
    if len(core) < MIN_CORE_FOR_SWAP:
        return rep
    lengths = {i: len(titles[i]) for i in core}
    median = statistics.median(lengths.values())
    if lengths[rep] < LONG_HEADLINE_RATIO * median:
        return rep
    terms = {i: _terms(titles[i]) for i in core}
    counts = Counter(t for ts in terms.values() for t in ts)
    key = {t for t, n in counts.items() if n >= max(2, KEY_TERM_SHARE * len(core))}

    def coverage(i: int) -> float:
        return len(key & terms[i]) / len(key) if key else 1.0

    shorter = [i for i in core if lengths[i] < lengths[rep] and headline_penalties(titles[i]) == 0
               and coverage(i) >= coverage(rep)]
    return min(shorter, key=lambda i: abs(lengths[i] - median)) if shorter else rep
