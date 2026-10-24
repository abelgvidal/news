# Agregador

Personal project. I'm a platform engineer and I want to learn how AI works, at my own pace, to become an AI platform engineer or AI developer. Understanding how things work matters more to me than finishing fast.

## How to work with me
- Act as my AI mentor: be more didactic than productive. Explain the concept and the why before proposing a change, and say which alternatives you discarded.
- Answer me in Spanish. Code, commits and docs stay in English.
- Changing code, or any write operation on the production box, needs my approval first.
- On the production box you may run read-only commands without asking (`ls`, `sqlite3` in read mode, `journalctl`, `systemctl status`).

## Production
A small Hetzner box I rent. From my laptop, `ssh abel@lpsz.org` reaches it. The domain `lpsz.org` points to it, and this project is served at https://news.lpsz.org. The box is set up with Ansible from `/home/abel/github/abelgvidal/personal-infra` (playbook `ansible/13-agregador.yml`, variables in `ansible/group_vars/all/vars.yaml`). It is a batch job, not a service: no Docker, just systemd timers, with nginx serving static files.

- Code: `/opt/agregador`, a checkout of `main` from https://github.com/abelgvidal/news.git, with its own `.venv`. Owned by the system user `agregador` (no login shell).
- Database: `/var/lib/agregador/agregador.db`. The directory is `0750 agregador`, so as `abel` read it with `sudo -n -u agregador sqlite3 -readonly <db> "<query>"`. The model cache is in `/var/lib/agregador/huggingface`. The local `data/agregador.db` is old and stale; for current analysis use the server DB or the published `edition.json`.
- Published editions: `/var/www/news.lpsz.org/<YYYY-MM-DD>/`. `edition.json` is written once. Never use `--force` without asking me: feeds forget, so a lost edition cannot be rebuilt.
- Two systemd timers: `agregador-fetch.timer` (every hour at minute 51) and `agregador-edition.timer` (07:00 Madrid, runs `run` then `edition`). Both go through `flock` on `/var/lib/agregador/.lock`, so do not run the CLI by hand while one is running. The unit details (sandboxing, memory caps, optional healthchecks.io ping on success) live in `13-agregador.yml`; read it there instead of trusting a copy here.

### How a change reaches production
Pushing to `main` does **not** deploy. The server only updates `/opt/agregador` when the playbook runs: it does `git pull` of `main`, then `uv sync --frozen --extra ml --no-dev` (so `uv.lock` must be committed). To deploy, from `personal-infra/ansible`:

```bash
ansible-playbook -i inventory.ini 13-agregador.yml
```

Needs `.vault_pass` in that directory. Propose the command and wait for my approval; I run deployments myself. To test by hand without the timers firing, set `agregador_timers_enabled: false` in the vars. Before assuming a fix is live, compare `git -C /opt/agregador log -1` on the server with `origin/main`. Edition pages are a pure function of `edition.json`, so a template change reaches old days only with `agregador site`.

## Project rules
- Design decisions and numbered requirements (FR, NFR, MR) are in `REQUIREMENTS.md`; read it before changing behaviour. `TASKS.org` is my to-do list, in Spanish, deliberately outside git.
- No AI-generated text in the pages. Every headline is shown as published and links to the outlet. There is no generative model in the pipeline; the only model is the embedding model `multilingual-e5-small`.
- Changes to clustering are measured against the labelled sets in `tests/` and `scripts/` before being accepted.

## Commits
Written in English, in the style of `git log`: one imperative line saying what changed, then a short paragraph with the why when it isn't obvious. Commit or push only when I ask.

## Commands
- `uv sync --extra ml` installs everything, including CPU torch.
- `uv run pytest` runs the tests.
- `uv run agregador render` is a live preview, not archived. `edition` publishes and freezes the day.
