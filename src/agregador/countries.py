"""Which country a story is about, from the headlines of its articles (a dictionary, no generative AI).

Each term of countries.toml found in a headline adds its weight to its country (a place 1.0, a person 0.5;
a term counts once per headline). The primary country is the one holding at least MIN_SHARE of all the weight,
provided there are at least MIN_MENTIONS of it: stories that mix countries (a summit between three
leaders) or mention none return None and are never limited.
"""
import re
import tomllib
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

PLACE_WEIGHT = 1.0
PERSON_WEIGHT = 0.5
MIN_MENTIONS = 2.0
MIN_SHARE = 0.5
INVOLVED_SHARE = 0.15  # a country with this share of the weight is involved in the story (flag shown)
MAX_INVOLVED = 4
_US = re.compile(r"\bU\.S\.A?\.?|\bUSA\b|\bUS\b")  # case-sensitive: the pronoun "us" is not the country
_UK = re.compile(r"\bU\.K\.?|\bUK\b")


@lru_cache(maxsize=1)
def _dictionary() -> tuple[re.Pattern, dict[str, tuple[str, float]]]:
    data = tomllib.loads((Path(__file__).parent / "countries.toml").read_text(encoding="utf-8"))
    terms: dict[str, tuple[str, float]] = {}
    for country, entry in data.items():
        for term in entry.get("places", []):
            terms[term] = (country, PLACE_WEIGHT)
        for term in entry.get("people", []):
            terms.setdefault(term, (country, PERSON_WEIGHT))
    longest_first = sorted(terms, key=len, reverse=True)  # "south africa" before "africa"
    pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in longest_first) + r")(?![a-z0-9])")
    return pattern, terms


@lru_cache(maxsize=1)
def _iso_codes() -> dict[str, str]:
    data = tomllib.loads((Path(__file__).parent / "countries.toml").read_text(encoding="utf-8"))
    return {country: entry["iso"] for country, entry in data.items() if entry.get("iso")}


def flag(country: str | None) -> str:
    """The flag emoji of a country of countries.toml (two regional-indicator letters); the globe if unknown."""
    code = _iso_codes().get(country or "")
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code) if code else "🌐"


def _fold(title: str) -> str:
    title = _UK.sub(" britain ", _US.sub(" united states ", title))
    return unicodedata.normalize("NFKD", title.lower()).encode("ascii", "ignore").decode()


def tally(titles: list[str]) -> Counter:
    """Weight of every country mentioned by the headlines."""
    pattern, terms = _dictionary()
    weight: Counter = Counter()
    for title in titles:
        for term in dict.fromkeys(pattern.findall(_fold(title))):  # each term once per headline, in order of appearance
            country, w = terms[term]
            weight[country] += w
    return weight


def involved_countries(titles: list[str]) -> list[str]:
    """The countries a story involves: those holding at least INVOLVED_SHARE of the weight, heaviest first.

    Ties keep the order in which the countries first appear. Empty when the headlines mention too few.
    """
    weight = tally(titles)
    total = sum(weight.values())
    if total < MIN_MENTIONS:
        return []
    return [c for c, w in weight.most_common() if w / total >= INVOLVED_SHARE][:MAX_INVOLVED]


def primary_country(titles: list[str]) -> str | None:
    """The country the story is about, or None when it is unclear or mixes several."""
    weight = tally(titles)
    total = sum(weight.values())
    if total < MIN_MENTIONS:
        return None
    country, top = weight.most_common(1)[0]
    return country if top / total >= MIN_SHARE else None
