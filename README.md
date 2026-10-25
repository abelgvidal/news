# news

A news aggregator that reads the front pages of a few dozen outlets, works out which of them are talking about the same thing, and prints the ten stories that most outlets agree are worth talking about. Once a day. Then it goes back to sleep.

It does not write anything. Every headline you see was written by a journalist, is shown exactly as published, and links to the outlet that wrote it. The software's contribution is counting.

## What it does (and refuses to do)

- **Reads RSS.** Public feeds only. No scraping, no logins, no browser pretending to be a person.
- **Groups the same story across outlets** with a small embedding model that runs on a CPU. No API, no key, no invoice.
- **Ranks by how many distinct outlets cover it**, with a half-life of 24 hours, so yesterday's scandal eventually has to leave.
- **Shows the real headline** of the most representative article. No summaries. No "AI-powered insights". No friendly paragraph explaining what you just read.
- **Two editions.** *International* (16 outlets, in English, untranslated) and *Spain* (17 outlets, in Spanish). Spanish world news is kept out of the Spain tab, because the International tab already has it and one of each is enough.
- **One story per country** in International. Otherwise it would be a daily report on the United States.
- **Padlocks** on outlets that tend to want your credit card.
- **Archive.** One frozen edition per day, with previous/next links. Yesterday's front page stays yesterday's front page.

## How a story is chosen

1. Each article is embedded (`intfloat/multilingual-e5-small`) and joined to the nearest cluster above a similarity threshold, or starts its own.
2. Small clusters that are really the same thing get merged. Big ones need much stronger evidence, because two big stories merging is how you get a headline about everything.
3. Inside a cluster, the medoid fixes *which* story it is. Articles that drifted into a neighbouring topic are dropped from the story and from the outlet count.
4. The representative headline is the most central one in that core, with small penalties for quotes, question marks, live blogs and headlines that are mostly capitals. Tie-breaker only. Clickbait is not banned, merely discouraged.

There is no generative model anywhere in the pipeline. The only machine learning is the embedding model, which turns text into numbers and has never had an opinion.

## Measured, not vibes

Clustering quality is the whole product, so it is evaluated against hand-checked sets in `tests/` and `scripts/`: labelled pairs, labelled clusters for both editions, and labelled foreign/domestic and country decisions. The criterion is deliberately strict: *the same specific news item*. Analysis, reactions and "related events" do not count, however related.

An honest list of what that does and does not prove:

- The first pass of the labels was made by a language model, then reviewed by a human. Not every label has been reviewed.
- Thresholds were tuned on one day of data. A second day is on the to-do list, next to "sleep".
- Cross-language clustering does not work well enough to use (correct matches and wrong ones overlap), so each edition is clustered in its own language.

## Run it

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra ml               # the "ml" extra pulls CPU-only torch
uv run agregador run             # fetch, purge, embed, cluster
uv run agregador edition         # publish today's edition into ./site
uv run pytest                    # 100-odd tests, a few seconds
```

Serve `site/` with anything that serves files. It is plain HTML, one stylesheet, and some thumbnails.

Other commands: `fetch`, `health` (which feeds are failing), `embed`, `cluster`, `recluster`, `clusters`, `explain <article id>` (why did this article end up there?), `render` (live preview, no images, not archived), `site` (rebuild all pages from the published editions).

## Configuration

Everything lives in `feeds.toml`: editions (`[[country]]`: language, time zone, whether to exclude foreign stories, one-per-country limit, flags) and feeds (`[[feed]]`: name, URL, edition, paywall flag, optional user agent for the one outlet whose firewall has opinions).

Adding an outlet is three lines. Deciding whether it deserves to be there is the hard part.

## Editions, archive and images

An edition lives in `site/<YYYY-MM-DD>/`: `edition.json` (written once, never regenerated), `img/` (thumbnails) and the pages. The pages are a pure function of `edition.json`, so `agregador site` can rebuild them all without touching what was published. The home page is the latest edition.

Story images are downloaded once per edition, shrunk to WebP thumbnails, and credited to the outlet they came from. Each image stays tied to its story: its filename carries the tab, rank and cluster id, its EXIF carries the headline and the source URLs, and the manifest records everything. Images are only fetched from public hosts, with size and pixel limits, and re-encoded so nothing from the original survives.

> **Heads up.** Serving a copy of someone else's image is a reproduction, not a link. If you run this publicly, get legal advice for your jurisdiction. This README is not legal advice. It is barely advice.

## Deploying

Collection and publishing are separate on purpose. Many feeds only keep the last few hours (half of them do not reach back 24 hours), so collecting once a day would miss most of the news.

- `agregador fetch` every hour, at minute 51: light, no model loaded.
- `agregador run` + `agregador edition` once a day at 07:00 Madrid time.

The timers are two systemd units: the hourly `fetch` (a `oneshot` service, `OnCalendar=*-*-* *:51:00`) and the daily edition (`OnCalendar=*-*-* 07:00:00 Europe/Madrid`, which runs `run` and then `edition`). State (the SQLite database) is cheap to lose; the published editions are not, because feeds forget. Back those up.

## Layout

```
src/agregador/   the pipeline: fetch, normalize, embed, cluster, representative, foreign, countries, render, edition, images
migrations/      SQLite schema, applied in order
feeds.toml       editions and outlets
scripts/         evaluation and labelling helpers
tests/           unit tests and the labelled sets
REQUIREMENTS.md  what it should do, and why
```

## License

[AGPL-3.0-or-later](LICENSE). If you run a modified version as a service, you share your changes. If you just read the code, nobody will know.

Headlines, images and trademarks belong to their publishers. This project links to them and counts them, which is rather the point.
