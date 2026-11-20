# Deadlock Meta Lab

Finds overtuned items, heroes, counters and hero-specific item fits at the current balance
patch, from deadlock-api match dumps + deadlock-data game files.

## Update / rebuild

```powershell
.\.venv\Scripts\python -m dla.ingest      # pull new matches for the patch window (incremental)
.\.venv\Scripts\python -m dla.load        # clean tables -> data/clean
.\.venv\Scripts\python -m dla.analyze     # models -> data/results   (--only heroes,items,counters,fits)
.\.venv\Scripts\python -m dla.dashboard   # -> data/dashboard/index.html
```

After changing only skill settings (brackets, `min_known_players`), `python -m dla.load --matches-only`
rebuilds the match table in about a minute instead of the full ~40-minute load.

After a balance patch: set `patch.patch_start` in `settings.yaml` (or `"auto"` to use the newest
changelog in deadlock-data), `git pull` deadlock-data, and run all four steps. Changing the window,
columns or filters makes `ingest` re-extract automatically.

## Layout

| Path | What |
|---|---|
| `settings.yaml` | every tunable: window, modes, columns, cleaning, rank brackets, model parameters |
| `config.yaml` | Steam credentials for deadbot only (not used by the pipeline; gitignored) |
| `dla/ingest.py` | projected/filtered remote reads of the ~750 GB `match_player` dump via DuckDB |
| `dla/gamedata.py` | deadlock-data loader; numeric ids = MurmurHash2(class name); validity rules |
| `dla/load.py` | dedupe, match validity, skill brackets (season-mapped badges), purchases, deaths |
| `dla/analysis/heroes.py` | hero stats, draft-model strength, hero counters & synergies |
| `dla/analysis/items.py` | game-state model and state-adjusted item / hero×item effects |
| `dla/analysis/counters.py` | item vs enemy hero: win-effect delta + kill-share odds |
| `dla/analysis/fits.py` | playstyle profiles and profile-predicted hero×item fit, underused flags |
| `dla/dashboard.py` | self-contained HTML dashboard |

## Data notes

- Ranked + Unranked, Normal mode only (Street Brawl excluded; 9999-cost items are Street Brawl only).
- Unranked matches carry no badge; their skill comes from players' own ranked games in the window.
- Beta Season 2 soft-reset ranks. In the dump it starts at the 2026-11-05 20:07–20:25 UTC downtime
  (`rank_interval` 1→2, ranked bans 3→4); hero balance had already shifted at ~05:00 that day.
  Season 1 badges are quantile-mapped onto the current Season 2 scale.
- ~40% of matches have no `banned_hero_ids` (missing, not zero); ban rates use matches that have one.
- The data store lives at `paths.data_dir` (currently `E:/dla-data`, ~20 GB raw + scratch).
- Corrupted items (temporary, pre-window) and Street Brawl item enhancements (`PropertyUpgrades`) are ignored.
- In the dump, `flags=1` marks an item consumed as a component of an upgrade (its sell time equals
  the parent's buy time in 99% of cases); a sell time with `flags=0` is a real sale.
