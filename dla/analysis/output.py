"""What changes in a player's own output after buying an item (and in the enemy's).

Win-rate effects of counter-buys are biased: they are bought from behind. This stage
measures the item's direct effect on what happens in fights instead, from the per-tick
stats (3-min snapshots to 15:00, then 5-min).

For a purchase made in stats interval k = (t[k-1], t[k]] we compare per-minute rates in
the interval before (k-1) with the interval after (k+1); the purchase interval itself is
skipped. Rates grow with time and souls for everyone, so a purchase is compared with the
average change of the same hero buying *any other item of the same tier in the same
interval* (the same baseline idea as the win effects). Effects are reported as the change
in per-minute rate and as % of the baseline's post-purchase rate.

Metrics (from stats.* and the per-tick damage split; CC applied is not recorded in the
data): hero damage dealt (total, bullet, ability), damage taken (total, bullet, ability),
healing, self healing, damage mitigated, enemy healing prevented, own healing lost to
anti-heal, kills, deaths.

Also:
  * by game state at purchase (behind / even / ahead from the state model's P(win)):
    win effect and damage-dealt effect per bucket, so items that only pay off when you
    are already winning (snowball) stand out from comeback / stabilising items;
  * per enemy hero: the change in the *enemy's* damage dealt and healing (and their healing
    lost) after you buy, against games without that hero (counter-buy effectiveness), and
    the change in your damage taken.

Runs on analysis.output.sample_fraction of matches (absolute share; must be <= the heavy
sample). Needs `scored` from items.run in the same session (P(win) at purchase).
Outputs: item_output, hero_item_output, item_state_split, item_vs_hero_output.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .common import Store, bh_fdr, scopes, z_pvalue

log = logging.getLogger("dla.analysis.output")

# name: (cumulative stats column, or per-interval delta column)
CUM = {"dmg_out": "stats_player_damage", "dmg_taken": "stats_player_damage_taken", "healing": "stats_player_healing",
       "self_heal": "stats_self_healing", "mitigated": "stats_damage_mitigated",
       "heal_prevented": "stats_heal_prevented", "heal_lost": "stats_heal_lost", "kills": "stats_kills",
       "deaths": "stats_deaths"}
DELTA = {"bullet_out": "dmg_bullet_out_delta", "ability_out": "dmg_ability_out_delta",
         "bullet_in": "dmg_bullet_in_delta", "ability_in": "dmg_ability_in_delta"}
ENEMY = ["dmg_out", "healing", "self_heal", "heal_lost"]          # enemy-side metrics for counters
BUCKETS = [(-1, 0.4, "behind"), (0.4, 0.6, "even"), (0.6, 2, "ahead")]


def build_intervals(st: Store, sample_pred: str, metrics: list[str]):
    """iv: per (match, slot, interval i) per-minute rates; the end-of-match interval is dropped."""
    have = {r[0] for r in st.con.execute("DESCRIBE players").fetchall()}
    cum = {m: c for m, c in CUM.items() if m in metrics and c in have}
    dlt = {m: c for m, c in DELTA.items() if m in metrics and c in have}
    missing = sorted(set(metrics) - set(cum) - set(dlt))
    if missing:
        log.warning("players.parquet lacks columns for %s (re-ingest with the new stats columns)", missing)
    un = ", ".join([f"unnest({c})::DOUBLE {m}" for m, c in cum.items()] + [f"unnest({c})::DOUBLE {m}" for m, c in dlt.items()])
    rate = ", ".join([f"({m} - lag({m}, 1, 0) OVER w) / (dt / 60) {m}" for m in cum] + [f"{m} / (dt / 60) {m}" for m in dlt])
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE iv AS
        WITH u AS (
          SELECT match_id, player_slot, team, generate_subscripts(stats_time_stamp_s, 1) i,
                 len(stats_time_stamp_s) n, unnest(stats_time_stamp_s)::INT t, {un}
          FROM players WHERE {sample_pred}
        ), d AS (SELECT *, t - lag(t, 1, 0) OVER (PARTITION BY match_id, player_slot ORDER BY i) dt FROM u)
        SELECT match_id, player_slot, team, i, t, {rate}
        FROM d WHERE i < n AND dt > 0
        WINDOW w AS (PARTITION BY match_id, player_slot ORDER BY i)
    """)
    return list(cum) + list(dlt)


def _bucket_sql(col="pe"):
    return "CASE " + " ".join(f"WHEN {col} < {hi} THEN '{name}'" for lo, hi, name in BUCKETS) + " END"


def _sums(metrics, prefix="d_", post=True):
    s = []
    for m in metrics:
        s += [f"sum({prefix}{m}) s_{m}", f"sum({prefix}{m} * {prefix}{m}) ss_{m}"]
        if post:
            s.append(f"sum(post_{m}) p_{m}")
    return ", ".join(s)


def purchase_deltas(st: Store, sample_pred: str, metrics: list[str]):
    """po: one row per purchase with pre/post deltas, keyed for aggregation."""
    d = ", ".join(f"b.{m} - a.{m} d_{m}, b.{m} post_{m}" for m in metrics)
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE po AS
        WITH pu AS (
          SELECT pu.match_id, pu.player_slot, pu.hero_id, pu.team, pu.item_id, pu.tier, pu.buy_t,
                 m.bracket, m.match_mode, m.in_scope
          FROM purchases pu JOIN matches m USING (match_id)
          WHERE m.in_scope AND {sample_pred.replace('match_id', 'pu.match_id')}
        ), k AS (
          SELECT pu.*, iv.i k FROM pu ASOF JOIN iv ON iv.match_id = pu.match_id AND iv.player_slot = pu.player_slot
                                              AND iv.t >= pu.buy_t
        )
        SELECT k.*, {_bucket_sql('coalesce(sc.pe, 0.5)')} bucket, {d}
        FROM k
        JOIN iv a ON a.match_id = k.match_id AND a.player_slot = k.player_slot AND a.i = k.k - 1
        JOIN iv b ON b.match_id = k.match_id AND b.player_slot = k.player_slot AND b.i = k.k + 1
        LEFT JOIN scored sc ON sc.match_id = k.match_id AND sc.player_slot = k.player_slot
                           AND sc.item_id = k.item_id AND sc.buy_t = k.buy_t
    """)
    return st.con.execute("SELECT count(*) FROM po").fetchone()[0]


def _effects(g: pd.DataFrame, base_keys: list[str], out_keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """g: summed rows keyed by base_keys + item_id (+ extra out keys). Effect of each item vs the
    pooled baseline of all items within base_keys, aggregated to out_keys."""
    tot = g.groupby(base_keys)[["n"] + [f"{p}_{m}" for m in metrics for p in ("s", "p")]].sum()
    tot.columns = [f"B{c}" for c in tot.columns]
    x = g.join(tot, on=base_keys)
    res = {}
    for m in metrics:
        b = x[f"Bs_{m}"] / x["Bn"]                         # baseline mean delta in this cell
        x[f"r_{m}"] = x[f"s_{m}"] - x["n"] * b              # sum of residuals
        x[f"rr_{m}"] = x[f"ss_{m}"] - 2 * b * x[f"s_{m}"] + x["n"] * b * b
        x[f"bp_{m}"] = x["n"] * x[f"Bp_{m}"] / x["Bn"]      # baseline post rate, weighted by this item's cells
    agg = x.groupby(out_keys)[["n"] + [f"{p}_{m}" for m in metrics for p in ("r", "rr", "bp")]].sum().reset_index()
    for m in metrics:
        n = agg.n
        mean = agg[f"r_{m}"] / n
        var = (agg[f"rr_{m}"] / n - mean ** 2).clip(lower=0)
        res[f"{m}"] = mean
        res[f"{m}_se"] = np.sqrt(var / n)
        res[f"{m}_pct"] = mean / (agg[f"bp_{m}"] / n).where(lambda v: v.abs() > 1e-9)
    return pd.concat([agg[out_keys + ["n"]], pd.DataFrame(res)], axis=1)


def _scope_filter(df, sc):
    if sc.bracket:
        df = df[df.bracket == sc.bracket]
    if sc.ranked_only:
        df = df[df.match_mode == "Ranked"]
    return df


def run(st: Store):
    s = st.s
    cfg = s["analysis"].get("output", {})
    frac = float(cfg.get("sample_fraction", 0.15))
    heavy = float(s["analysis"].get("heavy_sample_fraction", 1.0))
    if frac > heavy:
        log.warning("output.sample_fraction %.2f > heavy_sample_fraction %.2f; using the heavy sample", frac, heavy)
        frac = heavy
    pred = "TRUE" if frac >= 1 else f"(hash(match_id) % 10000) < {int(frac * 10000)}"
    metrics = build_intervals(st, pred, list(CUM) + list(DELTA))
    n = purchase_deltas(st, pred, metrics)
    log.info("output: %d purchases with pre/post windows, metrics %s", n, metrics)
    from ..gamedata import GameData
    g = GameData(s)
    items = g.items.reset_index()[["item_id", "name", "slot", "tier", "cost"]]
    heroes = g.heroes["name"].to_dict()
    keys = ["bracket", "match_mode", "hero_id", "tier", "k", "item_id", "bucket"]
    fine = st.q(f"SELECT {', '.join(keys)}, count(*) n, {_sums(metrics)} FROM po GROUP BY ALL")
    min_n = int(cfg.get("min_purchases", 300))
    min_hero_n = int(cfg.get("min_hero_purchases", 150))

    io, hio, split = [], [], []
    for sc in scopes(s):
        f = _scope_filter(fine, sc)
        if f.empty:
            continue
        cell = f.groupby(["hero_id", "tier", "k", "item_id"]).sum(numeric_only=True).reset_index()
        e = _effects(cell, ["hero_id", "tier", "k"], ["item_id"], metrics)
        e["scope"] = sc.name
        io.append(e[e.n >= min_n])
        he = _effects(cell, ["hero_id", "tier", "k"], ["hero_id", "item_id"], ["dmg_out", "dmg_taken", "healing"]
                      if "healing" in metrics else ["dmg_out", "dmg_taken"])
        he["scope"] = sc.name
        hio.append(he[he.n >= min_hero_n])
        cb = f.groupby(["hero_id", "tier", "k", "bucket", "item_id"]).sum(numeric_only=True).reset_index()
        sb = _effects(cb, ["hero_id", "tier", "k", "bucket"], ["item_id", "bucket"], ["dmg_out", "dmg_taken"])
        sb["scope"] = sc.name
        split.append(sb)
    item_output = pd.concat(io, ignore_index=True)
    for m in metrics:
        item_output[f"{m}_q"] = item_output.groupby("scope", group_keys=False)[[m, f"{m}_se"]].apply(
            lambda d: pd.Series(bh_fdr(z_pvalue(d[m], d[f"{m}_se"])), index=d.index))
    item_output = item_output.merge(items, on="item_id", how="left")
    hero_item_output = pd.concat(hio, ignore_index=True).merge(items[["item_id", "name", "tier"]], on="item_id", how="left")
    hero_item_output["hero"] = hero_item_output.hero_id.map(heroes)

    # win effect by state bucket at purchase (from the scored purchases of the heavy sample)
    wsplit = st.q(f"""
        WITH b AS (SELECT *, {_bucket_sql()} bucket FROM scored WHERE in_scope)
        SELECT bracket, match_mode, hero_id, tier, tick, bucket, item_id, count(*) n, sum(y - pe) r,
               sum(pe * (1 - pe)) v FROM b GROUP BY ALL
    """)
    ws = []
    for sc in scopes(s):
        f = _scope_filter(wsplit, sc)
        if f.empty:
            continue
        c = f.groupby(["hero_id", "tier", "tick", "bucket", "item_id"])[["n", "r", "v"]].sum().reset_index()
        base = c.groupby(["hero_id", "tier", "tick", "bucket"])[["n", "r"]].sum()
        c = c.join((base.r / base.n).rename("b"), on=["hero_id", "tier", "tick", "bucket"])
        c["rr"] = c.r - c.n * c.b
        a = c.groupby(["item_id", "bucket"])[["n", "rr", "v"]].sum().reset_index()
        a["win_effect"] = a.rr / a.n
        a["win_se"] = np.sqrt(a.v) / a.n
        a["scope"] = sc.name
        ws.append(a[["scope", "item_id", "bucket", "n", "win_effect", "win_se"]].rename(columns={"n": "win_n"}))
    state_split = pd.concat(split, ignore_index=True).merge(pd.concat(ws, ignore_index=True),
                                                           on=["scope", "item_id", "bucket"], how="outer")
    state_split = state_split.merge(items[["item_id", "name", "tier"]], on="item_id", how="left")

    st.write("item_output", item_output, meta={"metrics": metrics, "sample_fraction": frac, "purchases": n})
    st.write("hero_item_output", hero_item_output)
    st.write("item_state_split", state_split)
    del fine
    st.con.execute("DROP TABLE IF EXISTS po; DROP TABLE IF EXISTS iv")   # free memory for the enemy pass
    run_enemy(st, items, heroes)


def _items_heroes(s):
    from ..gamedata import GameData
    g = GameData(s)
    return g.items.reset_index()[["item_id", "name", "slot", "tier", "cost"]], g.heroes["name"].to_dict()


def run_enemy(st: Store, items=None, heroes=None):
    """item_vs_hero_output on its own match sample (analysis.output.enemy_sample_fraction).

    Memory-lean: each player's pre/post change per interval is computed once (eiv), and the
    purchase x enemy-player join (~6 rows per purchase) is streamed straight into a GROUP BY
    rather than materialized. Needs no scored purchases, so it can run alone
    (python -m dla.analyze --only output_enemy)."""
    s = st.s
    cfg = s["analysis"].get("output", {})
    if items is None:
        items, heroes = _items_heroes(s)
    frac = float(cfg.get("enemy_sample_fraction", 0.1))
    pred = "TRUE" if frac >= 1 else f"(hash(match_id) % 10000) < {int(frac * 10000)}"
    want = list(dict.fromkeys(ENEMY + ["dmg_taken", "bullet_in", "ability_in"]))
    metrics = build_intervals(st, pred, want)
    em = [m for m in ENEMY if m in metrics]
    own = [m for m in ("dmg_taken", "bullet_in", "ability_in") if m in metrics]
    win = "OVER (PARTITION BY match_id, player_slot ORDER BY i)"
    dd = ", ".join(f"lead({m}) {win} - lag({m}) {win} d_{m}, lead({m}) {win} post_{m}" for m in metrics)
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE eiv AS
        WITH w AS (SELECT match_id, player_slot, team, i, lag(i) {win} pi, lead(i) {win} ni, {dd} FROM iv)
        SELECT w.* EXCLUDE (pi, ni), p.hero_id
        FROM w JOIN players p ON p.match_id = w.match_id AND p.player_slot = w.player_slot
        WHERE w.pi = w.i - 1 AND w.ni = w.i + 1 AND {pred.replace('match_id', 'p.match_id')}
    """)
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pk AS
        WITH pu AS (
          SELECT pu.match_id, pu.player_slot, pu.team, pu.item_id, pu.tier, pu.buy_t,
                 m.bracket, m.match_mode
          FROM purchases pu JOIN matches m USING (match_id)
          WHERE m.in_scope AND {pred.replace('match_id', 'pu.match_id')}
        ), k AS (
          SELECT pu.*, iv.i k FROM pu ASOF JOIN iv ON iv.match_id = pu.match_id AND iv.player_slot = pu.player_slot
                                              AND iv.t >= pu.buy_t
        )
        SELECT k.match_id, k.team, k.k, k.item_id, k.tier, k.bracket, k.match_mode,
               {', '.join(f'e.d_{m}, e.post_{m}' for m in own)}
        FROM k JOIN eiv e ON e.match_id = k.match_id AND e.player_slot = k.player_slot AND e.i = k.k
    """)
    st.con.execute("DROP TABLE IF EXISTS iv")
    n = st.con.execute("SELECT count(*) FROM pk").fetchone()[0]
    log.info("enemy output: %d purchases (sample %.2f)", n, frac)
    sums = ", ".join([f"sum(pk.d_{m}) s_{m}, sum(pk.d_{m} * pk.d_{m}) ss_{m}, sum(pk.post_{m}) p_{m}" for m in own] +
                     [f"sum(e.d_{m}) s_e_{m}, sum(e.d_{m} * e.d_{m}) ss_e_{m}, sum(e.post_{m}) p_e_{m}" for m in em])
    w = st.q(f"""
        SELECT pk.bracket, pk.match_mode, pk.tier, pk.k, pk.item_id, e.hero_id enemy_hero, count(*) n, {sums}
        FROM pk JOIN eiv e ON e.match_id = pk.match_id AND e.i = pk.k AND e.team <> pk.team
        GROUP BY ALL
    """)
    allown = st.q(f"SELECT bracket, match_mode, tier, k, item_id, count(*) n, {_sums(own)} FROM pk GROUP BY ALL")
    st.con.execute("DROP TABLE IF EXISTS pk; DROP TABLE IF EXISTS eiv")
    df = enemy_output(st, w, allown, own, em, items, heroes, cfg)
    st.write("item_vs_hero_output", df, meta={"enemy_sample_fraction": frac, "purchases": n})


def enemy_output(st: Store, w, allown, own, em, items, heroes, cfg) -> pd.DataFrame:
    """Per (item, enemy hero): your damage-taken change and the enemy's own output change after
    you buy, each vs purchases of the same tier/interval with that enemy present, minus the
    same item's effect in games without that enemy."""
    ecols = [f"e_{m}" for m in em]
    stat_cols = lambda ms: ["n"] + [f"{p}_{m}" for m in ms for p in ("s", "ss", "p")]
    out = []
    for sc in scopes(st.s):
        f = _scope_filter(w, sc)
        if f.empty:
            continue
        f = f.groupby(["tier", "k", "item_id", "enemy_hero"]).sum(numeric_only=True).reset_index()
        withb = _effects(f, ["tier", "k", "enemy_hero"], ["item_id", "enemy_hero"], own + ecols)
        # own metrics without B: all purchases of the item minus those facing B
        a = _scope_filter(allown, sc).groupby(["tier", "k", "item_id"])[stat_cols(own)].sum().reset_index()
        rows = []
        for hb, fb in f.groupby("enemy_hero"):
            wo = a.merge(fb[["tier", "k", "item_id"] + stat_cols(own)], on=["tier", "k", "item_id"], how="left",
                         suffixes=("", "_b")).fillna(0)
            for c in stat_cols(own):
                wo[c] = wo[c] - wo[f"{c}_b"]
            wo = wo[["tier", "k", "item_id"] + stat_cols(own)]
            wo["enemy_hero"] = hb
            rows.append(wo[wo.n > 0])
        without = _effects(pd.concat(rows, ignore_index=True), ["tier", "k", "enemy_hero"], ["item_id", "enemy_hero"], own)
        # enemy metrics: the item's average effect over all enemies (general, not hero-specific)
        fe = f.groupby(["tier", "k", "item_id"])[stat_cols(ecols)].sum().reset_index()
        general = _effects(fe, ["tier", "k"], ["item_id"], ecols)
        x = withb.merge(without.drop(columns="n"), on=["item_id", "enemy_hero"], suffixes=("", "_wo"))
        x = x.merge(general.drop(columns="n"), on="item_id", suffixes=("", "_wo"))
        for m in own + ecols:
            x[f"{m}_delta"] = x[m] - x[f"{m}_wo"]
            x[f"{m}_delta_se"] = np.sqrt(x[f"{m}_se"] ** 2 + x[f"{m}_se_wo"] ** 2)
            x[f"{m}_delta_pct"] = x[f"{m}_pct"] - x[f"{m}_pct_wo"]
        x["scope"] = sc.name
        out.append(x[x.n >= int(cfg.get("min_enemy_purchases", 300))])
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not df.empty:
        df = df.merge(items[["item_id", "name", "tier"]], on="item_id", how="left")
        df["enemy"] = df.enemy_hero.map(heroes)
    return df
