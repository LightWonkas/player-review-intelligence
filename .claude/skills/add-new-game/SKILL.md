---
name: add-new-game
description: Adds a mobile game to config/apps.yaml, checks that its Google Play package and App Store id are real and belong to that game, runs a small fetch and reports what came back. Use when the user wants to track another game or gives store links for one.
argument-hint: "[game name or store URLs]"
---

Add the game described in `$ARGUMENTS` to `config/apps.yaml`.

## 1. Collect the details

Read the existing entry in `config/apps.yaml` for the format. You need:

- `slug`: kebab-case, unique, e.g. `space-arena`
- `name` and `developer`
- **Google Play** `store_app_id`: the package name, from `?id=` in a Play URL (e.g. `com.example.game`)
- **App Store** `store_app_id`: the number after `id` in an App Store URL, as a quoted string
- `country` / `language`: default `us` / `en` unless the user says otherwise

A game may have only one store. If an id is missing, ask the user for the store link. Never guess an id from the game's name.

## 2. Check the ids belong to the game

App Store:
```bash
curl -s "https://itunes.apple.com/lookup?id=<id>&country=<country>"
```
Expect `"resultCount":1`. Show `trackName` and `sellerName`.

Google Play:
```bash
uv run python -c "from google_play_scraper import app; a = app('<package>', lang='<language>', country='<country>'); print(a['title'], '|', a['developer'], '|', a['installs'])"
```

If a lookup fails, or the title or developer doesn't match the game the user means, stop and show what came back.

## 3. Add the entry

Append an entry under `games:` in the same shape as the existing one, then validate it:
```bash
uv run review-intel db init
```
This fails with a clear message on a bad id format or a duplicate slug. Fix the entry and run it again.

## 4. Small fetch

```bash
uv run review-intel fetch --app <slug> --store all --limit 50
```

## 5. Report

Report reviews fetched per store, plus the date range and rating mix, from:
```bash
uv run python -c "import sqlite3; c = sqlite3.connect('data/review_intel.db'); print(c.execute(\"select a.store, count(*), min(date(r.review_date)), max(date(r.review_date)), round(avg(r.rating), 2) from reviews r join apps a on a.id = r.app_id where a.slug = '<slug>' group by a.store\").fetchall())"
```

Then suggest the next steps, with the cost estimate from
`uv run review-intel classify --app <slug> --dry-run --limit 1000`:
classify, `patches fetch` (App Store only), `index`, and the `weekly-review-report` skill.

## Rules

- Don't commit unless the user asks. If they do, use a `feat(config): track <game>` commit.
- Keep README prose game-agnostic: the game's name goes in `config/apps.yaml`, not the README.
- Respect store rate limits: one lookup per store, and keep the first fetch small.
