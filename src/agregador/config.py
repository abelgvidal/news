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


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    tz: str = "UTC"


def _load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_feeds(path: Path) -> list[FeedConfig]:
    return [FeedConfig(**entry) for entry in _load(path).get("feed", [])]


def load_countries(path: Path) -> list[Country]:
    """Countries in file order; the first is the home page."""
    return [Country(**entry) for entry in _load(path).get("country", [])]
