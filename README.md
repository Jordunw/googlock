# Deadlock Meta Lab

Finds overtuned items, heroes, counters and hero-specific item fits at the current balance
patch, from deadlock-api match dumps + deadlock-data game files.

## Update / rebuild

```powershell
.\.venv\Scripts\python -m dla.ingest      # pull new matches for the patch window (incremental)
.\.venv\Scripts\python -m dla.load        # clean tables -> data/clean
.\.venv\Scripts\python -m dla.analyze     # models -> data/results   (--only heroes,items,counters,output,fits,value; output_enemy reruns just the per-enemy part)
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
| `dla/analysis/output.py` | output change after a purchase (damage dealt/taken, healing, mitigation, deaths), win effect by game state at purchase, enemy output change per enemy hero |
| `dla/itemvalue.py` | item value from game data only: card properties -> effective HP / healing / damage / CC -> souls, ROI vs the shop's stat items |
| `dla/itemvalue.js` | the same engine for the browser: the Item value tab recomputes live (hero picker, editable assumptions); keep in step with itemvalue.py |
| `dla/ability_rules.yaml` | how hero abilities become spirit damage / CC / healing per fight for the item-value kit model: generic reading + per-ability rules where the data alone is wrong (Mina's ult, Vyper's Venom, ...) |
| `dla/item_rules.yaml` | what each item-card property means (generic) + per-item overrides for mechanics the card does not express |
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
- Core items (bought by >= `analysis.items.core_buy_rate` of a hero's players) get a CORE badge and an
  on-time landmark effect instead of the win delta, which is ~0 for them by construction.
- Per-tick damage is split into bullet / ability from the engagement-distance custom stats (per-interval
  increments, `dmg_*_delta` columns). Crowd control applied is not recorded anywhere in the dump.
- Counterbuy planner (dashboard tab): computed in the page from the item-counter, matchup and hero-item tables;
  defaults and the team-stacking item list are in `settings.yaml` (`counterbuy`), all adjustable on the tab.
- Item value (`python -m dla.itemvalue` or `--only value`) needs no match data; tune `settings.yaml`/`item_value`.
  It writes `item_value_inputs.json` (item cards, per-hero stats, rules, defaults), which the dashboard embeds and
  `itemvalue.js` evaluates live. After changing either engine run `python -m dla.itemvalue --check` (needs node):
  it compares every number of the two. Each hero's kit (casts, spirit damage and its scaling, CC) is read from
  its abilities; the Item value tab lists it per ability, and `dla/ability_rules.yaml` corrects abilities the
  generic reading gets wrong. Rule/settings edits: rerun the value stage, then `dla.dashboard`.
