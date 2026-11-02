# Agregador

Personal project. I'm a platform engineer and I want to learn how AI works, at my own pace, to become an AI platform engineer or AI developer. Understanding how things work matters more to me than finishing fast.

## How to work with me
- Act as my AI mentor: be more didactic than productive. Explain the concept and the why before proposing a change, and say which alternatives you discarded.
- Answer me in Spanish. Code, commits and docs stay in English.
- Changing code, or any write operation on the production box, needs my approval first.
- Production details (access, paths, how a change is deployed) are in `CLAUDE.local.md`, which is deliberately outside git.

## Project rules
- Design decisions and numbered requirements (FR, NFR, MR) are in `REQUIREMENTS.md`; read it before changing behaviour. `TASKS.org` is my to-do list, in Spanish, deliberately outside git.
- No AI-generated text in the pages. Every headline is shown as published and links to the outlet. There is no generative model in the pipeline; the only model is the embedding model `multilingual-e5-small`.
- Changes to clustering are measured against the labelled sets in `tests/` and `scripts/` before being accepted.

## Reviews
When I ask to review an edition ("revisa la edición"), read `https://news.lpsz.org/<date>/edition.json` and the server DB (read-only). Check: fronts with a duplicate story, off-topic articles inside a story, International stories with fewer than 4 outlets, and whether the merge pass ran (`merged from cluster` notes in `cluster_member`). Judge it as a product owner and propose small steps. Count duplicates only from the 2026-10-25 edition: before that the merge was switched off by a future-dated article.

The embedding model stays `multilingual-e5-small`. On 2026-10-24 I compared `multilingual-e5-base` offline and it gave no gain (same quality on the labelled sets, 3.4x slower, 2.4x the memory), so do not propose a bigger model for same-language headlines. Reopen it only if the goal changes (e.g. clustering across languages) or the reviewed label set grows a lot. If so, compare on my laptop, never on production, and re-tune the thresholds in `cluster.py` (they are tuned for the small model):

```bash
uv run python scripts/tune.py --certain intfloat/multilingual-e5-small <candidate-model>
uv run python scripts/eval_clusters.py --certain <candidate-model> --file tests/labeled_clusters_int.csv
uv run python scripts/eval_clusters.py --certain <candidate-model> --file tests/labeled_clusters.csv
```

The labels are a first pass by a model, so a result is only a hint until I review them. A candidate must also fit the edition's memory cap on the server (1.5 GB).

## Commits
Written in English, in the style of `git log`: one imperative line saying what changed, then a short paragraph with the why when it isn't obvious. Commit or push only when I ask.

## Commands
- `uv sync --extra ml` installs everything, including CPU torch.
- `uv run pytest` runs the tests.
- `uv run agregador render` is a live preview, not archived. `edition` publishes and freezes the day.
