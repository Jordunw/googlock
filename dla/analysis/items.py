"""Item effects adjusted for game state.

Raw item win rates are dominated by selection: players who are already ahead buy more and
more expensive items. We therefore:

1. Fit a *state model*: P(team wins | game state at tick t), where state = souls lead
   (absolute and relative), kill lead, skill difference, draft strength difference
   (from hero_strength), side and mode. One logistic model with per-tick coefficients
   (ticks are the ~3-minute stats snapshots), so it is linear and can be evaluated in SQL
   for every purchase.
2. For every purchase, residual r = won - P(win | state at the last tick before purchase).
3. Item effect = mean residual of its buyers minus the mean residual of all purchases of
   the same tier at the same tick (the "bought something else of this price" baseline),
   so effects are relative to the average item of that tier. Optionally per hero
   (baseline also per hero), shrunk toward the item's global effect.

Outputs: item_effects (per scope), hero_item_effects (per scope), state_model (coefs +
calibration).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ..gamedata import GameData
from .common import Scope, Store, bh_fdr, eb_shrink_effect, scopes, z_pvalue

log = logging.getLogger("dla.analysis.items")

STATE_FEATS = ["lead_k", "lead_rel", "kill_diff", "skill_diff", "draft_diff", "side", "side_ranked"]


def build_state(st: Store, max_tick: int):
    """Team-0 perspective state per (match, tick). Tick 0 = pre-game (only draft/skill)."""
    strength = pd.read_parquet(st.results / "hero_strength.parquet")
    strength = strength[strength.scope == "All"][["hero_id", "strength_logit"]]
    st.con.register("strength_df", strength)
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE draft AS
        SELECT p.match_id,
               sum(CASE WHEN p.team = 'Team0' THEN s.strength_logit ELSE -s.strength_logit END) draft_diff
        FROM players_s p LEFT JOIN strength_df s USING (hero_id) GROUP BY 1
    """)
    # Stats snapshots: every 3 min to 15:00, then every 5 min, plus one at match end. The
    # end-of-match point is dropped (it leaks the result); the rest are the state ticks.
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE state AS
        WITH tk AS (
          SELECT match_id, team, unnest(stats_time_stamp_s)::INT t, unnest(stats_net_worth)::DOUBLE nw,
                 unnest(stats_kills)::DOUBLE k, generate_subscripts(stats_time_stamp_s, 1) i,
                 len(stats_time_stamp_s) n_ticks
          FROM players_s
        ), ag AS (
          SELECT match_id, t,
                 sum(CASE WHEN team = 'Team0' THEN nw ELSE -nw END)::DOUBLE lead,
                 sum(nw)::DOUBLE total,
                 sum(CASE WHEN team = 'Team0' THEN k ELSE -k END)::DOUBLE kill_diff
          FROM tk WHERE i < n_ticks AND t <= {max_tick} AND t % 60 = 0
          GROUP BY ALL HAVING count(*) = 12
        ), zero AS (SELECT match_id, 0 t, 0.0 lead, 0.0 total, 0.0 kill_diff FROM matches_s)
        SELECT a.match_id, a.t, a.lead / 1000 lead_k, a.lead / greatest(a.total, 1) lead_rel, a.kill_diff,
               coalesce(m.skill_step_diff, 0) skill_diff, coalesce(d.draft_diff, 0) draft_diff,
               1.0 side, (m.match_mode = 'Ranked')::DOUBLE side_ranked,
               (m.winning_team = 'Team0')::INT y0, m.in_scope, m.bracket, m.match_mode
        FROM (SELECT * FROM ag UNION ALL SELECT * FROM zero) a
        JOIN matches_s m USING (match_id) LEFT JOIN draft d USING (match_id)
    """)


def fit_state_model(st: Store, sample_matches: int, seed: int = 0):
    df = st.q(f"""
        SELECT * FROM state WHERE in_scope AND match_id IN (
          SELECT match_id FROM matches_s WHERE in_scope USING SAMPLE reservoir({int(sample_matches)} ROWS) REPEATABLE ({seed}))
    """)
    rng = np.random.default_rng(seed)
    test_ids = set(rng.choice(df.match_id.unique(), size=max(1, df.match_id.nunique() // 5), replace=False))
    is_test = df.match_id.isin(test_ids).to_numpy()
    coefs, calib = [], []
    for t, g in df.groupby("t"):
        tr, te = g[~is_test[g.index]], g[is_test[g.index]]
        if len(tr) < 2000:
            continue
        # symmetric augmentation: mirror to Team1 perspective (side terms flip sign)
        X = tr[STATE_FEATS].to_numpy()
        Xa = np.vstack([X, -X])
        ya = np.concatenate([tr.y0, 1 - tr.y0])
        mdl = LogisticRegression(C=10.0, fit_intercept=False, max_iter=1000).fit(Xa, ya)
        c = dict(zip(STATE_FEATS, mdl.coef_[0]), t=t, n_train=len(tr))
        coefs.append(c)
        if len(te):
            p = expit(te[STATE_FEATS].to_numpy() @ mdl.coef_[0])
            calib.append(dict(t=t, n_test=len(te), auc=roc_auc_score(te.y0, p) if te.y0.nunique() > 1 else np.nan,
                              log_loss=log_loss(te.y0, p, labels=[0, 1]), mean_p=p.mean(), mean_y=te.y0.mean()))
    coefs = pd.DataFrame(coefs)
    log.info("state model calibration:\n%s", pd.DataFrame(calib).round(3).to_string())
    return coefs, pd.DataFrame(calib)


def score_purchases(st: Store, coefs: pd.DataFrame):
    st.con.register("coef_df", coefs)
    lin = " + ".join(f"s.{f} * c.{f}" for f in STATE_FEATS)
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE scored AS
        SELECT pu.match_id, pu.player_slot, pu.hero_id, pu.team, pu.won::INT y, pu.item_id, pu.tier, pu.cost,
               pu.buy_t, pu.removed_how, s.t tick,
               CASE WHEN pu.team = 'Team0' THEN 1 / (1 + exp(-({lin}))) ELSE 1 / (1 + exp({lin})) END pe,
               m.bracket, m.match_mode, m.in_scope
        FROM purchases_s pu
        JOIN matches_s m USING (match_id)
        ASOF JOIN state s ON s.match_id = pu.match_id AND s.t <= pu.buy_t
        JOIN coef_df c ON c.t = s.t
    """)
    st.con.execute("UPDATE scored SET pe = least(greatest(pe, 1e-4), 1 - 1e-4)")


def _effects(st: Store, sc: Scope, by_hero: bool, min_n: int) -> pd.DataFrame:
    w = sc.where.replace("m.", "")
    hero_g = "hero_id," if by_hero else ""
    df = st.q(f"""
        WITH base AS (
          SELECT {hero_g} tier, tick, avg(y - pe) b FROM scored WHERE {w} GROUP BY ALL
        ), r AS (
          SELECT s.*, (s.y - s.pe) - base.b AS rr FROM scored s
          JOIN base USING ({hero_g} tier, tick) WHERE {w.replace('in_scope', 's.in_scope')}
        )
        SELECT {hero_g} item_id, count(*) n, avg(y) raw_win_rate, avg(pe) expected_win_rate,
               avg(rr) effect, sqrt(sum(pe * (1 - pe))) / count(*) effect_se,
               median(buy_t) / 60 median_buy_min,
               avg(coalesce(removed_how = 'sold', false)::INT) sold_rate, avg(coalesce(removed_how = 'upgraded', false)::INT) upgraded_rate
        FROM r GROUP BY ALL
    """)
    df["p_value"] = z_pvalue(df.effect, df.effect_se)
    df["q_value"] = bh_fdr(df.p_value)
    df["low_sample"] = df.n < min_n
    df["scope"] = sc.name
    return df


def run(st: Store):
    s = st.s
    cfg = s["analysis"]["items"]
    g = GameData(s)
    max_tick = cfg["max_tick_min"] * 60
    build_state(st, max_tick)
    coefs, calib = fit_state_model(st, cfg["state_model_sample_matches"])
    st.write("state_model_coefs", coefs)
    st.write("state_model_calibration", calib)
    score_purchases(st, coefs)

    items = g.items.reset_index()[["item_id", "name", "slot", "tier", "cost", "activation", "component_ids"]]
    heroes = g.heroes["name"].to_dict()
    all_item, all_hero_item = [], []
    for sc in scopes(s):
        log.info("scope %s", sc.name)
        ie = _effects(st, sc, False, cfg["min_purchases"])
        if ie.empty:
            continue
        ie["effect_per_1k_souls"] = ie.effect / ie.item_id.map(items.set_index("item_id").cost) * 1000
        all_item.append(ie)

        he = _effects(st, sc, True, cfg["min_hero_purchases"])
        # hierarchical shrinkage: hero x item deviation from the item's global effect
        he = he.merge(ie[["item_id", "effect"]].rename(columns={"effect": "item_effect"}), on="item_id")
        dev, dev_sd, tau = eb_shrink_effect(he.effect - he.item_effect, he.effect_se)
        he["hero_specific"] = dev                       # how much better on this hero than on average
        he["hero_specific_sd"] = dev_sd
        he["effect_shrunk"] = he.item_effect + dev
        # usage: share of this hero's players who buy the item
        usage = st.q(f"""
            SELECT hero_id, count(*) players FROM players_s p JOIN matches_s m USING (match_id)
            WHERE {sc.where} GROUP BY 1
        """)
        he = he.merge(usage, on="hero_id")
        he["buy_rate"] = he.n / he.players
        all_hero_item.append(he)

        # breadth: across heroes with enough purchases, share with a positive shrunk effect
        ok = he[~he.low_sample]
        br = ok.groupby("item_id").agg(heroes_n=("hero_id", "nunique"),
                                       heroes_positive=("effect_shrunk", lambda x: float((x > 0).mean())),
                                       hero_mean_effect=("effect_shrunk", "mean")).reset_index()
        all_item[-1] = all_item[-1].merge(br, on="item_id", how="left")

    ie = pd.concat(all_item, ignore_index=True).merge(items, on="item_id", how="left")
    he = pd.concat(all_hero_item, ignore_index=True).merge(items[["item_id", "name", "slot", "tier", "cost"]],
                                                           on="item_id", how="left")
    he["hero"] = he.hero_id.map(heroes)
    ie["component_ids"] = ie.component_ids.map(list)
    st.write("item_effects", ie)
    st.write("hero_item_effects", he)
