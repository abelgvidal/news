import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeedConfig:
    name: str
    url: str
    country: str = "es"
    weight: float = 1.0
    user_agent: str | None = None  # override for the few feeds whose firewall rejects our default one
    paywall: bool = False  # articles often need a subscription: shown with a lock


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    tz: str = "UTC"
    lang: str = "es"  # language of the page: labels and dates (the headlines are shown as published)
    exclude_foreign: bool = False  # keep out stories about the rest of the world (see foreign.py)
    max_per_country: int = 0  # at most this many stories about the same country (0 = no limit; see countries.py)
    show_flags: bool = False  # a flag (or a globe when there is no single country) on each story


def _load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_feeds(path: Path) -> list[FeedConfig]:
    return [FeedConfig(**entry) for entry in _load(path).get("feed", [])]


def load_countries(path: Path) -> list[Country]:
    """Countries in file order; the first is the home page."""
    return [Country(**entry) for entry in _load(path).get("country", [])]
