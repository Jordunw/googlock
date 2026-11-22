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

Core items. An item almost every player of a hero buys forms most of its own baseline,
so its effect is ~0 by construction (Burst Fire on Vyper: 93% buy rate, +0.4 pp). Items
with buy_rate >= items.core_buy_rate on a hero are flagged `core` and measured instead by
a landmark comparison: at the first state tick at/after the item's median buy time on that
hero, players who already own it (or anything built from it) vs players who don't, each
against P(win | state at that tick). That is the value of having it on time vs being late
or skipping it, not of the item vs another item.

Outputs: item_effects (per scope), hero_item_effects (per scope, with `core`),
hero_core_items (landmark effects), state_model (coefs + calibration).
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


def _team_pe(lin: str) -> str:
    return f"CASE WHEN x.team = 'Team0' THEN 1 / (1 + exp(-({lin}))) ELSE 1 / (1 + exp({lin})) END"


def descendants(items: pd.DataFrame) -> dict[int, set[int]]:
    """item -> itself plus every item that has it (recursively) as a component."""
    parents: dict[int, set[int]] = {}
    for iid, comps in zip(items.item_id, items.component_ids):
        for c in comps:
            parents.setdefault(c, set()).add(iid)
    out = {}
    for iid in items.item_id:
        seen, todo = {iid}, [iid]
        while todo:
            for p in parents.get(todo.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    todo.append(p)
        out[iid] = seen
    return out


def core_items(st: Store, he: pd.DataFrame, items: pd.DataFrame, coefs: pd.DataFrame) -> pd.DataFrame:
    """Landmark on-time effect for every (hero, item) flagged core in any scope."""
    core = he[he.core]
    if core.empty:
        return pd.DataFrame()
    ticks = np.array(sorted(coefs.t))
    # landmark per (hero, item): median buy time in the pooled scope (or wherever it is core)
    med = he[he.scope == "All"].set_index(["hero_id", "item_id"]).median_buy_min
    keys = core[["hero_id", "item_id"]].drop_duplicates()
    m = [med.get((h, i), np.nan) for h, i in zip(keys.hero_id, keys.item_id)]
    keys = keys.assign(med_t=np.array(m, float) * 60)
    keys["L"] = [int(ticks[min(np.searchsorted(ticks, max(t, 1)), len(ticks) - 1)]) if np.isfinite(t) else None
                 for t in keys.med_t]
    keys = keys.dropna(subset=["L"]).astype({"L": int})
    desc = descendants(items)
    sat = pd.DataFrame([(i, d) for i in keys.item_id.unique() for d in desc.get(i, {i})],
                       columns=["item_id", "sat_item_id"])
    st.con.register("core_df", keys[["hero_id", "item_id", "L"]])
    st.con.register("sat_df", sat)
    st.con.register("coef_df", coefs)
    lin = " + ".join(f"s.{f} * c.{f}" for f in STATE_FEATS)
    # Materialized in steps: the single multi-join query trips a DuckDB 2.0 optimizer assertion
    # ("dict-surviving join ... build pipeline is not single-source") on the full dataset.
    st.con.execute("""
        CREATE OR REPLACE TEMP TABLE core_b AS
        SELECT pu.match_id, pu.player_slot, s.item_id, min(pu.buy_t) fb
        FROM purchases_s pu JOIN sat_df s ON s.sat_item_id = pu.item_id GROUP BY ALL
    """)
    st.con.execute(f"""
        CREATE OR REPLACE TEMP TABLE core_x AS
        SELECT p.match_id, p.player_slot, p.hero_id, p.won::INT y, m.bracket::VARCHAR bracket,
               m.match_mode::VARCHAR match_mode, k.item_id, k.L, {_team_pe(lin).replace('x.team', 'p.team')} pe
        FROM players_s p JOIN matches_s m USING (match_id)
        JOIN core_df k ON k.hero_id = p.hero_id
        JOIN state s ON s.match_id = p.match_id AND s.t = k.L
        JOIN coef_df c ON c.t = s.t
        WHERE m.in_scope
    """)
    agg = st.q("""
        SELECT x.bracket, x.match_mode, x.hero_id, x.item_id, x.L,
               coalesce(b.fb <= x.L, false) has, count(*) n, sum(x.y) sy, sum(x.pe) spe,
               sum(x.pe * (1 - x.pe)) v
        FROM core_x x
        LEFT JOIN core_b b ON b.match_id = x.match_id AND b.player_slot = x.player_slot AND b.item_id = x.item_id
        GROUP BY ALL
    """)
    st.con.execute("DROP TABLE IF EXISTS core_b; DROP TABLE IF EXISTS core_x")
    out = []
    for sc in scopes(st.s):
        a = agg
        if sc.bracket:
            a = a[a.bracket == sc.bracket]
        if sc.ranked_only:
            a = a[a.match_mode == "Ranked"]
        if a.empty:
            continue
        gsum = a.groupby(["hero_id", "item_id", "L", "has"])[["n", "sy", "spe", "v"]].sum().reset_index()
        w = gsum.pivot_table(index=["hero_id", "item_id", "L"], columns="has", values=["n", "sy", "spe", "v"],
                             fill_value=0)
        w.columns = [f"{c}_{'has' if h else 'not'}" for c, h in w.columns]
        w = w.reset_index()
        for c in ["n", "sy", "spe", "v"]:
            for h in ["has", "not"]:
                if f"{c}_{h}" not in w:
                    w[f"{c}_{h}"] = 0
        n1, n0 = w.n_has.clip(lower=1), w.n_not.clip(lower=1)
        w["on_time_rate"] = w.n_has / (w.n_has + w.n_not)
        w["wr_has"], w["wr_not"] = w.sy_has / n1, w.sy_not / n0
        w["on_time_effect"] = (w.sy_has - w.spe_has) / n1 - (w.sy_not - w.spe_not) / n0
        w["on_time_se"] = np.sqrt(w.v_has / n1 ** 2 + w.v_not / n0 ** 2)
        w.loc[(w.n_has < 50) | (w.n_not < 50), ["on_time_effect", "on_time_se"]] = np.nan
        w["q_value"] = bh_fdr(z_pvalue(w.on_time_effect, w.on_time_se))
        w["scope"] = sc.name
        out.append(w.drop(columns=[c for c in w.columns if c.startswith(("sy_", "spe_", "v_"))]))
    df = pd.concat(out, ignore_index=True).rename(columns={"L": "landmark_s"})
    # keep only pairs that are core in that scope
    flags = he[["scope", "hero_id", "item_id", "core", "buy_rate", "median_buy_min"]]
    df = df.merge(flags, on=["scope", "hero_id", "item_id"], how="inner")
    return df[df.core].drop(columns="core")


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
        he["core"] = he.buy_rate >= cfg.get("core_buy_rate", 0.5)
        all_hero_item.append(he)

        # breadth: across heroes with enough purchases, share with a positive shrunk effect
        ok = he[~he.low_sample]
        br = ok.groupby("item_id").agg(heroes_n=("hero_id", "nunique"),
                                       heroes_positive=("effect_shrunk", lambda x: float((x > 0).mean())),
                                       hero_mean_effect=("effect_shrunk", "mean")).reset_index()
        br = br.merge(he.groupby("item_id").core.sum().rename("core_heroes").reset_index(), on="item_id", how="outer")
        all_item[-1] = all_item[-1].merge(br, on="item_id", how="left")

    ie = pd.concat(all_item, ignore_index=True).merge(items, on="item_id", how="left")
    he = pd.concat(all_hero_item, ignore_index=True).merge(items[["item_id", "name", "slot", "tier", "cost"]],
                                                           on="item_id", how="left")
    he["hero"] = he.hero_id.map(heroes)
    ie["component_ids"] = ie.component_ids.map(list)
    st.write("item_effects", ie)
    st.write("hero_item_effects", he)
    log.info("core items: landmark analysis for %d (scope, hero, item) pairs", int(he.core.sum()))
    ci = core_items(st, he, items, coefs)
    if not ci.empty:
        ci = ci.merge(items[["item_id", "name", "slot", "tier", "cost"]], on="item_id", how="left")
        ci["hero"] = ci.hero_id.map(heroes)
    st.write("hero_core_items", ci)
