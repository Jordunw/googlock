"""Item <-> enemy-hero counters.

Two independent signals; a real hard counter should show up in both.

item_vs_hero_win   For item X and enemy hero B: state-adjusted effect of X in matches where B
                   is on the enemy team minus its effect where B is not. ("Phantom Strike is
                   worth +4% more when the enemy has Vindicta.")
item_vs_hero_kill  Kill-share ratio. Among kills made by players *holding* X (bought, not yet
                   sold/upgraded) in matches where B is an enemy, the share whose victim is B,
                   divided by the same share for kills by non-holders, stratified by match
                   and 5-minute time bucket (Mantel-Haenszel). >1 = holders of X kill B
                   disproportionately often (offensive counter).
                   Defensive version: among deaths of players holding X, share caused by B vs
                   non-holders. <1 = X protects against B.

Both computed once with (bracket, match_mode) as keys and rolled up to scopes.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..gamedata import GameData
from .common import Store, bh_fdr, eb_shrink_effect, scopes, z_pvalue

log = logging.getLogger("dla.analysis.counters")


def _rollup(df: pd.DataFrame, keys: list[str], sums: list[str], st: Store) -> pd.DataFrame:
    parts = []
    for sc in scopes(st.s):
        d = df
        if sc.bracket:
            d = d[d.bracket == sc.bracket]
        if sc.ranked_only:
            d = d[d.match_mode == "Ranked"]
        a = d.groupby(keys, as_index=False)[sums].sum()
        a["scope"] = sc.name
        parts.append(a)
    return pd.concat(parts, ignore_index=True)


def item_vs_hero_win(st: Store, min_tier: int, min_n: int) -> pd.DataFrame:
    """Requires the `scored` table from items.run (same session).

    Effect of item X with enemy B present vs absent, where *each side has its own
    baseline*: X is compared with other items of the same tier bought at the same tick in
    matches with B as an enemy (resp. without B). Otherwise anything that shifts every
    residual in B's matches (e.g. a hero the state model under-rates late) would show up
    as every item "countering" B."""
    # residual sums at (bracket, mode, item, tier, tick[, enemy]) grain; everything else in pandas
    fine_e = st.q(f"""
        SELECT s.bracket, s.match_mode, s.item_id, s.tier, s.tick, e.hero_id enemy_hero,
               count(*) n, sum(s.y - s.pe) s_res, sum(s.pe * (1 - s.pe)) s_var
        FROM scored s JOIN players_s e ON e.match_id = s.match_id AND e.team <> s.team
        WHERE s.in_scope AND s.tier >= {min_tier}
        GROUP BY ALL
    """)
    fine = st.q(f"""
        SELECT bracket, match_mode, item_id, tier, tick, count(*) n, sum(y - pe) s_res, sum(pe * (1 - pe)) s_var
        FROM scored WHERE in_scope AND tier >= {min_tier} GROUP BY ALL
    """)
    out = []
    for sc in scopes(st.s):
        fe, fa = fine_e, fine
        if sc.bracket:
            fe, fa = fe[fe.bracket == sc.bracket], fa[fa.bracket == sc.bracket]
        if sc.ranked_only:
            fe, fa = fe[fe.match_mode == "Ranked"], fa[fa.match_mode == "Ranked"]
        fe = fe.groupby(["item_id", "tier", "tick", "enemy_hero"], as_index=False)[["n", "s_res", "s_var"]].sum()
        fa = fa.groupby(["item_id", "tier", "tick"], as_index=False)[["n", "s_res", "s_var"]].sum()
        # baselines per (tier, tick): with enemy B, and without B
        bw = fe.groupby(["tier", "tick", "enemy_hero"], as_index=False)[["n", "s_res"]].sum()
        ba = fa.groupby(["tier", "tick"], as_index=False)[["n", "s_res"]].sum()
        bw = bw.merge(ba, on=["tier", "tick"], suffixes=("", "_all"))
        bw["base_with"] = bw.s_res / bw.n
        bw["base_wo"] = (bw.s_res_all - bw.s_res) / (bw.n_all - bw.n).clip(lower=1)
        x = fe.merge(fa, on=["item_id", "tier", "tick"], suffixes=("", "_all"))
        x = x.merge(bw[["tier", "tick", "enemy_hero", "base_with", "base_wo"]], on=["tier", "tick", "enemy_hero"])
        x["n_wo"] = x.n_all - x.n
        x["rr_with"] = x.s_res - x.n * x.base_with
        x["rr_wo"] = (x.s_res_all - x.s_res) - x.n_wo * x.base_wo
        x["var_wo"] = x.s_var_all - x.s_var
        g = x.groupby(["item_id", "enemy_hero"], as_index=False)[["n", "n_wo", "rr_with", "rr_wo", "s_var", "var_wo"]].sum()
        g["effect_with"] = g.rr_with / g.n
        g["effect_without"] = g.rr_wo / g.n_wo.clip(lower=1)
        g["delta"] = g.effect_with - g.effect_without
        g["delta_se"] = np.sqrt(g.s_var / g.n ** 2 + g.var_wo / g.n_wo.clip(lower=1) ** 2)
        g["delta_shrunk"], g["delta_shrunk_sd"], _ = eb_shrink_effect(g.delta, g.delta_se)
        g["p_value"] = z_pvalue(g.delta, g.delta_se)
        g["q_value"] = bh_fdr(g.p_value)
        g["scope"] = sc.name
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df["low_sample"] = df.n < min_n
    return df.drop(columns=["rr_with", "rr_wo", "s_var", "var_wo", "n_wo"])


def item_vs_hero_kill(st: Store, min_tier: int, bucket_s: int, min_kills: int, holder: str) -> pd.DataFrame:
    """holder='killer' (offensive) or 'victim' (defensive). Mantel-Haenszel odds ratio of
    'the other party is hero B' for holders vs non-holders of X, stratified by match x time bucket."""
    me, other = ("killer_slot", "victim_hero") if holder == "killer" else ("victim_slot", "killer_hero")
    # the holder's side: killers are on the victim's opposite team
    team_expr = ("CASE d.victim_team WHEN 'Team0' THEN 'Team1' ELSE 'Team0' END"
                 if holder == "killer" else "d.victim_team")
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dk AS
        SELECT d.match_id, d.{me} slot, d.{other} other_hero, d.t, m.bracket, m.match_mode,
               -- stratum = match x side of the holder x time bucket
               (d.t // {bucket_s}) * 2 + ({team_expr} = 'Team0')::INT tb
        FROM deaths_s d JOIN matches_s m USING (match_id)
        WHERE m.in_scope AND d.killer_hero IS NOT NULL AND d.killer_slot <> d.victim_slot
    """)
    # inventory holding intervals (tier >= min_tier); one row per (match, slot, item) interval
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE hold AS
        SELECT match_id, player_slot slot, item_id, buy_t, coalesce(sold_t, 1e9) end_t
        FROM purchases_s WHERE tier >= {min_tier}
    """)
    # per stratum (match, time bucket) and B: deaths with other=B / total, split by holder status of X.
    # MH odds ratio: sum(a*d/n) / sum(b*c/n) where a = holder&B, b = holder&notB, c = non&B, d = non&notB
    df = st.q(f"""
        WITH ev AS (   -- every death event x every item: is this party holding it?
          SELECT k.match_id, k.tb, k.bracket, k.match_mode, k.other_hero, h.item_id
          FROM dk k JOIN hold h ON h.match_id = k.match_id AND h.slot = k.slot AND k.t >= h.buy_t AND k.t < h.end_t
        ),
        strata AS (    -- deaths per stratum and other hero (all parties)
          SELECT match_id, tb, other_hero, count(*) n_b FROM dk GROUP BY ALL
        ),
        strata_tot AS (SELECT match_id, tb, any_value(bracket) bracket, any_value(match_mode) match_mode,
                              count(*) n FROM dk GROUP BY ALL),
        held AS (      -- holder counts per stratum, item, other hero
          SELECT match_id, tb, item_id, other_hero, count(*) a FROM ev GROUP BY ALL
        ),
        held_tot AS (SELECT match_id, tb, item_id, count(*) h FROM ev GROUP BY ALL),
        cells AS (
          SELECT ht.match_id, ht.tb, ht.item_id, s.other_hero, st.bracket, st.match_mode, st.n, ht.h,
                 coalesce(hd.a, 0) a, s.n_b
          FROM held_tot ht
          JOIN strata_tot st USING (match_id, tb)
          JOIN strata s USING (match_id, tb)
          LEFT JOIN held hd ON hd.match_id = ht.match_id AND hd.tb = ht.tb AND hd.item_id = ht.item_id
                           AND hd.other_hero = s.other_hero
          WHERE st.n > ht.h          -- stratum needs both holders and non-holders
        )
        SELECT bracket, match_mode, item_id, other_hero,
               sum(a) a, sum(h - a) b, sum(n_b - a) c,
               sum(a * ((n - h) - (n_b - a)) / n) mh_num, sum((h - a) * (n_b - a) / n) mh_den,
               -- Robins-Breslow-Greenland variance pieces
               sum((a + ((n - h) - (n_b - a))) / n * a * ((n - h) - (n_b - a)) / n) p_r,
               sum(((a + ((n - h) - (n_b - a))) / n * (h - a) * (n_b - a) / n
                    + ((h - a) + (n_b - a)) / n * a * ((n - h) - (n_b - a)) / n)) ps_qr,
               sum(((h - a) + (n_b - a)) / n * (h - a) * (n_b - a) / n) q_s
        FROM cells GROUP BY ALL
    """)
    sums = ["a", "b", "c", "mh_num", "mh_den", "p_r", "ps_qr", "q_s"]
    df = _rollup(df, ["item_id", "other_hero"], sums, st)
    # an empty cell makes the MH odds ratio 0 or inf; leave those undefined (they are low-sample)
    R, S = df.mh_num.where(df.mh_num > 0), df.mh_den.where(df.mh_den > 0)
    df["odds_ratio"] = R / S
    df["log_or_se"] = np.sqrt(df.p_r / (2 * R ** 2) + df.ps_qr / (2 * R * S) + df.q_s / (2 * S ** 2))
    df["log_or"] = np.log(df.odds_ratio)
    out = []
    for sc, g in df.groupby("scope"):
        g = g.copy()
        g["log_or_shrunk"], g["log_or_shrunk_sd"], _ = eb_shrink_effect(g.log_or, g.log_or_se)
        g["p_value"] = z_pvalue(g.log_or, g.log_or_se)
        g["q_value"] = bh_fdr(g.p_value)
        out.append(g)
    df = pd.concat(out)
    df["holder_events"] = df.a
    df["low_sample"] = df.a < min_kills
    df["direction"] = "offensive" if holder == "killer" else "defensive"
    return df.drop(columns=["mh_num", "mh_den", "p_r", "ps_qr", "q_s"]).rename(columns={"other_hero": "enemy_hero"})


def run(st: Store, scored_ready: bool = True):
    s = st.s
    cfg = s["analysis"]["counters"]
    g = GameData(s)
    items = g.items[["name", "tier", "slot"]]
    heroes = g.heroes["name"]

    def label(df):
        df["item"] = df.item_id.map(items["name"])
        df["tier"] = df.item_id.map(items["tier"])
        df["enemy"] = df.enemy_hero.map(heroes)
        return df

    if scored_ready:
        st.write("item_vs_hero_win", label(item_vs_hero_win(st, cfg["min_tier"], cfg["min_purchases"])))
    kills = pd.concat([
        item_vs_hero_kill(st, cfg["min_tier"], cfg["time_bucket_s"], cfg["min_kill_events"], "killer"),
        item_vs_hero_kill(st, cfg["min_tier"], cfg["time_bucket_s"], cfg["min_kill_events"], "victim"),
    ])
    st.write("item_vs_hero_kill", label(kills))
