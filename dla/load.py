"""Build clean analysis tables from the raw extracts.

Outputs (data/clean/*.parquet):
  players    one row per (match, player) for valid matches, + skill estimate
  matches    one row per match: winner, duration, mode, skill, bans
  purchases  one row per shop-item purchase (valid items only), with souls at purchase
  abilities  ability skill-up events (ability unlocks / points) for build-order analysis
  accounts   per-account rank estimate from Ranked games (used as skill for Unranked)
"""
from __future__ import annotations

import argparse
import logging
import shutil

import duckdb
import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .gamedata import GameData

log = logging.getLogger("dla.load")


class QuantileMap:
    """Monotone map from one season's badge-step scale to the reference (latest) season's,
    by matching the quantiles of the same accounts' badges in both seasons."""

    def __init__(self, src=None, dst=None, n_overlap=0):
        self.src, self.dst, self.n = src, dst, n_overlap

    def __call__(self, v):
        if self.src is None or v is None or np.isnan(v):
            return v
        return float(np.interp(v, self.src, self.dst))

    def describe(self):
        if self.src is None:
            return dict(n_overlap=self.n, identity=True)
        mid = float(np.interp(np.median(self.src), self.src, self.dst) - np.median(self.src))
        return dict(n_overlap=self.n, identity=False, median_shift_steps=mid,
                    src_q=list(map(float, self.src)), dst_q=list(map(float, self.dst)))


def _apply_maps(maps: dict, season: pd.Series, step: pd.Series) -> np.ndarray:
    out = step.to_numpy(float).copy()
    se = season.to_numpy()
    for k, m in maps.items():
        if m.src is not None:
            sel = se == k
            out[sel] = np.interp(out[sel], m.src, m.dst)
    return out


def season_maps(acc: pd.DataFrame, min_overlap: int = 500) -> dict[int, QuantileMap]:
    """Maps every season onto the latest one (the current in-game scale)."""
    ref = int(acc.season.max())
    maps = {ref: QuantileMap()}
    base = acc[acc.season == ref].set_index("account_id").step
    for se in sorted(set(acc.season) - {ref}):
        cur = acc[acc.season == se].set_index("account_id").step
        both = cur.index.intersection(base.index)
        if len(both) < min_overlap:
            log.warning("season %d: only %d overlapping accounts; using identity map", se, len(both))
            maps[se] = QuantileMap(n_overlap=len(both))
            continue
        q = np.linspace(0, 1, 101)
        src, dst = np.quantile(cur[both], q), np.quantile(base[both], q)
        src, idx = np.unique(src, return_index=True)  # interp needs increasing x
        maps[se] = QuantileMap(src, dst[idx], len(both))
        log.info("season %d -> %d map: %d overlapping accounts, median shift %.2f steps", se, ref, len(both),
                 maps[se].describe()["median_shift_steps"])
    return maps


def _season_expr(s: Settings) -> str:
    starts = sorted(s["skill"].get("season_starts") or [])
    return " + ".join(f"(start_time >= TIMESTAMPTZ '{x}')::INT" for x in starts) or "0"


def build_matches(con, s: Settings, out):
    """One row per match, with skill bracket from the mean known skill of its players.
    Needs a `players` relation; cheap, so it can be rebuilt alone (--matches-only) after
    changing skill settings."""
    sk = s["skill"]
    bracket_case = "CASE " + " ".join(
        f"WHEN skill_tier BETWEEN {lo} AND {hi} THEN '{name}'" for name, (lo, hi) in sk["brackets"].items()) + " END"
    tmp = (out / "matches_tmp.parquet").as_posix()
    con.execute(f"""
        COPY (
          WITH m AS (
            SELECT match_id, any_value(start_time) start_time, any_value(duration_s) duration_s,
                   any_value(match_mode) match_mode, any_value(winning_team) winning_team,
                   any_value(average_badge) average_badge,
                   any_value(average_badge_team0) badge_team0, any_value(average_badge_team1) badge_team1,
                   avg(skill_step) skill_step_mean, count(skill_step) skill_known,
                   avg(skill_step) FILTER (WHERE team = 'Team0') - avg(skill_step) FILTER (WHERE team = 'Team1')
                       AS skill_step_diff,
                   any_value(banned_hero_ids) banned_hero_ids,
                   list(hero_id ORDER BY player_slot) FILTER (WHERE team = 'Team0') heroes_team0,
                   list(hero_id ORDER BY player_slot) FILTER (WHERE team = 'Team1') heroes_team1
            FROM players GROUP BY match_id
          )
          SELECT *, {_season_expr(s)} AS season,
                 CASE WHEN skill_known >= {int(sk['min_known_players'])} THEN floor(skill_step_mean / 6)::INT END
                     AS skill_tier,
          FROM m
        ) TO '{tmp}' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (SELECT *, {bracket_case} AS bracket,
                     coalesce(skill_tier >= {int(sk['exclude_below_tier'])}, true) AS in_scope
              FROM '{tmp}')
        TO '{(out / 'matches.parquet').as_posix()}' (FORMAT parquet)
    """)
    (out / "matches_tmp.parquet").unlink()
    log.info("matches written: %s", con.execute(f"""
        SELECT list(struct_pack(mode := match_mode, bracket := coalesce(bracket, 'none'), n := n) ORDER BY match_mode, bracket)
        FROM (SELECT match_mode, bracket, count(*) n FROM '{(out / 'matches.parquet').as_posix()}' GROUP BY ALL)""").fetchone()[0])


def build(s: Settings):
    raw = (s.data_dir / "raw" / s["ingest"]["table"]).as_posix() + "/*.parquet"
    out = s.data_dir / "clean"
    out.mkdir(exist_ok=True)
    c = s["clean"]
    g = GameData(s)

    db = s.data_dir / "work.duckdb"
    db.unlink(missing_ok=True)
    con = duckdb.connect(db.as_posix())  # file-backed so large intermediates can spill
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{c.get('memory_limit', '16GB')}'")
    con.execute(f"SET temp_directory='{(s.data_dir / 'duckdb_tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    if c.get("threads"):
        con.execute(f"SET threads={int(c['threads'])}")
    items = g.items.reset_index()[["item_id", "key", "name", "slot", "tier", "cost", "valid", "is_imbue"]]
    con.register("item_dim", items)
    abil = g.abilities.reset_index()[["ability_id", "key", "hero_id", "slot"]]
    con.register("abil_dim", abil)

    # 1. de-duplicate: the same match can appear in several dump files (base/delta/residual).
    #    A light key-only pass picks one file per match (the one with most rows); then each
    #    raw file is filtered to its picked matches and staged on its own, so the heavy
    #    nested columns are only ever streamed, never hashed or materialised in memory.
    stage = s.data_dir / "stage" / "dedup"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    con.execute(f"""
        CREATE OR REPLACE TABLE pick AS
        SELECT match_id, arg_max(filename, n) AS filename
        FROM (SELECT match_id, filename, count(*) n
              FROM read_parquet('{raw}', union_by_name=true, filename=true) GROUP BY ALL)
        GROUP BY match_id
    """)
    files = [r[0] for r in con.execute("SELECT DISTINCT filename FROM pick ORDER BY 1").fetchall()]
    for i, f in enumerate(files):
        con.execute(f"""
            COPY (SELECT * FROM read_parquet('{f}')
                  WHERE match_id IN (SELECT match_id FROM pick WHERE filename = '{f}'))
            TO '{(stage / f"{i:05d}.parquet").as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """)
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            log.info("dedup: staged %d/%d files", i + 1, len(files))
    con.execute(f"CREATE OR REPLACE VIEW p AS SELECT * FROM read_parquet('{stage.as_posix()}/*.parquet', union_by_name=true)")
    dup = con.execute("SELECT count(*) - count(DISTINCT (match_id, player_slot)) FROM p").fetchone()[0]
    if dup:
        raise RuntimeError(f"{dup} duplicate (match, slot) rows inside single raw files")
    # 2. match validity
    conds = [f"min(duration_s) >= {int(c['min_duration_s'])}", "count(*) = 12",
             "count(DISTINCT team) = 2"]
    if c.get("drop_abandoned_matches"):
        conds.append("max(abandon_match_time_s) = 0")
    if c.get("drop_low_pri"):
        conds.append("NOT bool_or(coalesce(low_pri_pool, false))")
    if c.get("drop_new_player_pool"):
        conds.append("NOT bool_or(coalesce(new_player_pool, false))")
    con.execute(f"""
        CREATE OR REPLACE TABLE good AS
        SELECT match_id FROM p GROUP BY match_id HAVING {' AND '.join(conds)}
    """)
    n_all = con.execute("select count(distinct match_id) from p").fetchone()[0]
    n_ok = con.execute("select count(*) from good").fetchone()[0]
    log.info("matches: %d raw, %d valid (%.1f%%)", n_all, n_ok, 100 * n_ok / max(n_all, 1))

    # 3. skill. Badges are put on a contiguous scale (6 sub-ranks per tier:
    #    step = tier*6 + subrank-1), split by ranked season, and earlier seasons are mapped
    #    onto the latest (current) season's scale by quantile matching over accounts seen
    #    in both, so brackets mean what the in-game badge means today.
    season_expr = _season_expr(s)
    step = "((player_rank_initial_display_rank // 10) * 6 + (player_rank_initial_display_rank % 10) - 1)"
    con.execute(f"""
        CREATE OR REPLACE TABLE ranked_obs AS
        SELECT account_id, match_id, player_slot, {season_expr} AS season, {step}::DOUBLE AS step
        FROM p WHERE match_mode = 'Ranked' AND player_rank_initial_display_rank > 0
    """)
    acc = con.execute("""
        SELECT account_id, season, median(step) AS step, count(*) AS games FROM ranked_obs GROUP BY ALL
    """).df()
    maps = season_maps(acc)
    ref = max(maps)
    log.info("ranked observations per season: %s (reference scale: season %d)",
             acc.groupby("season").games.sum().to_dict(), ref)
    acc["step_ref"] = _apply_maps(maps, acc.season, acc.step)
    # account skill: prefer reference-season (current) observations, else a mapped older season
    acc = acc.sort_values(["account_id", "season"], ascending=[True, False])
    accounts = acc.groupby("account_id").agg(step_ref=("step_ref", "first"), ranked_games=("games", "sum"),
                                             seasons=("season", "nunique")).reset_index()
    con.register("accounts_df", accounts)
    con.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM accounts_df")
    con.execute(f"COPY accounts TO '{(out / 'accounts.parquet').as_posix()}' (FORMAT parquet)")
    # per-match mapped badge for ranked games
    obs = con.execute("SELECT match_id, player_slot, season, step FROM ranked_obs").df()
    obs["step_ref"] = _apply_maps(maps, obs.season, obs.step)
    con.register("obs_df", obs[["match_id", "player_slot", "step_ref"]])
    pd.DataFrame([dict(season=k, **m.describe()) for k, m in maps.items()]).to_parquet(out / "season_maps.parquet")

    # 4. players (drop the heavy nested columns; they go to purchases/abilities)
    heavy = {"items_item_id", "items_game_time_s", "items_sold_time_s", "items_flags", "items_upgrade_id",
             "items_imbued_ability_id"}
    cols = [r[0] for r in con.execute("describe p").fetchall() if r[0] not in heavy]
    con.execute(f"""
        COPY (
          SELECT {', '.join('p.' + x for x in cols)}, a.step_ref AS account_step,
                 a.ranked_games AS account_ranked_games, coalesce(o.step_ref, a.step_ref) AS skill_step
          FROM p JOIN good USING (match_id)
          LEFT JOIN accounts a USING (account_id)
          LEFT JOIN obs_df o USING (match_id, player_slot)
        ) TO '{(out / 'players.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    con.execute(f"CREATE OR REPLACE VIEW players AS SELECT * FROM '{(out / 'players.parquet').as_posix()}'")
    log.info("players written")

    # 5. matches
    build_matches(con, s, out)

    # 6. purchases: unnest item events, keep shop items, attach souls at purchase time.
    #    Item events (shop items and ability skill-ups share the list) are streamed from the
    #    staged rows each time rather than materialised (~1e9 rows).
    log.info("matches written")
    ev = """(
        SELECT p.match_id, p.account_id, p.player_slot, p.hero_id, p.team, p.won,
               unnest(items_item_id) AS id, unnest(items_game_time_s) AS t,
               unnest(items_sold_time_s) AS sold_t, unnest(items_imbued_ability_id) AS imbued_ability_id,
               unnest(items_flags) AS flags,
               p.stats_time_stamp_s AS ts, p.stats_net_worth AS nw
        FROM p JOIN good USING (match_id))"""
    # souls (net worth) at purchase: linear interpolation on the stats ticks,
    # with an implicit (t=0, souls=0) point before the first tick.
    con.execute(f"""
        COPY (
          WITH ev AS {ev}, x AS (
            SELECT e.*, d.name, d.slot, d.tier, d.cost,
                   len(list_filter(e.ts, lambda v: v <= e.t)) AS i
            FROM ev e JOIN item_dim d ON d.item_id = e.id AND d.valid
          ), y AS (
            SELECT *,
                   CASE WHEN i = 0 THEN 0 ELSE ts[i] END::DOUBLE AS ta,
                   CASE WHEN i = 0 THEN 0 ELSE nw[i] END::DOUBLE AS na,
                   coalesce(ts[i + 1], ts[i])::DOUBLE AS tb, coalesce(nw[i + 1], nw[i])::DOUBLE AS nb
            FROM x
          )
          SELECT match_id, account_id, player_slot, hero_id, team, won, id AS item_id, name, slot, tier, cost,
                 t AS buy_t, nullif(sold_t, 0) AS sold_t, nullif(imbued_ability_id, 0) AS imbued_ability_id,
                 -- flags=1 = consumed as a component when the upgraded item was bought (verified:
                 -- 99% coincide with a buy of a parent item at the same second); a sell time with
                 -- flags=0 = sold back to the shop.
                 CASE WHEN sold_t > 0 AND flags = 1 THEN 'upgraded'
                      WHEN sold_t > 0 THEN 'sold' END AS removed_how,
                 CASE WHEN tb > ta THEN na + (nb - na) * (t - ta) / (tb - ta) ELSE na END AS souls_at_buy
          FROM y
        ) TO '{(out / 'purchases.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)

    log.info("purchases written")
    # 7. ability skill-ups (for build-order / ability-point analysis)
    con.execute(f"""
        COPY (
          WITH ev AS {ev}
          SELECT e.match_id, e.player_slot, e.hero_id, e.won, e.id AS ability_id, a.slot AS ability_slot, e.t
          FROM ev e JOIN abil_dim a ON a.ability_id = e.id
        ) TO '{(out / 'abilities.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    log.info("abilities written")
    # 8. deaths: who killed whom and when (killer slot -> hero via a small slot map)
    con.execute("CREATE OR REPLACE TABLE slot_hero AS SELECT match_id, player_slot, hero_id FROM players")
    con.execute(f"""
        COPY (
          WITH d AS (
            SELECT match_id, player_slot AS victim_slot, hero_id AS victim_hero, team AS victim_team,
                   unnest(death_details_game_time_s) AS t, unnest(death_details_killer_player_slot) AS killer_slot,
                   unnest(death_details_death_duration_s) AS death_duration_s
            FROM players
          )
          SELECT d.*, k.hero_id AS killer_hero
          FROM d LEFT JOIN slot_hero k ON k.match_id = d.match_id AND k.player_slot = d.killer_slot
        ) TO '{(out / 'deaths.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    con.close()
    shutil.rmtree(s.data_dir / "stage", ignore_errors=True)
    db.unlink(missing_ok=True)
    con = duckdb.connect()
    for t in ["players", "matches", "purchases", "abilities", "accounts", "deaths"]:
        n = con.execute(f"select count(*) from '{(out / (t + '.parquet')).as_posix()}'").fetchone()[0]
        log.info("%-10s %12d rows", t, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings")
    ap.add_argument("--matches-only", action="store_true",
                    help="only rebuild matches.parquet (brackets) from the existing players.parquet")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = load_settings(a.settings)
    if a.matches_only:
        out = s.data_dir / "clean"
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        con.execute(f"SET memory_limit='{s['clean'].get('memory_limit', '16GB')}'")
        con.execute(f"SET temp_directory='{(s.data_dir / 'duckdb_tmp').as_posix()}'")
        con.execute(f"CREATE VIEW players AS SELECT * FROM '{(out / 'players.parquet').as_posix()}'")
        build_matches(con, s, out)
    else:
        build(s)


if __name__ == "__main__":
    main()
