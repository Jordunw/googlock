"""Hero strength, counters and synergies.

hero_stats      descriptive pick/ban/win rates, KDA, souls, WR by game length (per scope)
hero_strength   L2 logistic regression of match outcome on team compositions, adjusted
                for skill difference, side, and mode. Coefficients = hero strength in
                log-odds independent of who they were drafted with/against and player skill.
hero_counters   for hero A vs enemy hero B: mean(actual - expected win) over matches where
                they face each other, where expected comes from hero_strength. Positive =
                A beats B more than their individual strengths predict. EB-shrunk.
hero_synergy    same for A and B on the same team.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from ..gamedata import GameData
from .common import Scope, Store, bh_fdr, eb_shrink_effect, eb_shrink_rate, scopes, wilson, z_pvalue

log = logging.getLogger("dla.analysis.heroes")

DURATION_BUCKETS = [0, 25 * 60, 35 * 60, 45 * 60, 10 ** 6]
DURATION_LABELS = ["<25m", "25-35m", "35-45m", "45m+"]


def hero_stats(st: Store, sc: Scope, min_games: int) -> pd.DataFrame:
    n_matches = st.q(f"SELECT count(*) n FROM matches m WHERE {sc.where}").n[0]
    df = st.q(f"""
        SELECT p.hero_id, count(*) games, sum(p.won::INT) wins,
               avg(p.kills) kills, avg(p.deaths) deaths, avg(p.assists) assists,
               avg(p.net_worth / (m.duration_s / 60.0)) souls_per_min,
               avg(p.final_melee_kills::DOUBLE / greatest(p.kills, 1)) melee_kill_share,
               avg(p.final_ability_kills::DOUBLE / greatest(p.kills, 1)) ability_kill_share,
               avg(p.final_bullet_kills::DOUBLE / greatest(p.kills, 1)) bullet_kill_share
        FROM players p JOIN matches m USING (match_id) WHERE {sc.where} GROUP BY 1
    """)
    # ~40% of matches in the dump carry no ban list (missing data, not "no bans"), so ban
    # rates are over matches that have one.
    n_ban_matches = st.q(f"SELECT count(*) n FROM matches m WHERE {sc.where} AND len(banned_hero_ids) > 0").n[0]
    bans = st.q(f"""
        SELECT h AS hero_id, count(*) bans FROM (
          SELECT unnest(banned_hero_ids) h FROM matches m WHERE {sc.where}) GROUP BY 1
    """)
    dur = st.q(f"""
        SELECT p.hero_id,
               CASE {' '.join(f"WHEN m.duration_s < {hi} THEN '{lab}'" for hi, lab in zip(DURATION_BUCKETS[1:], DURATION_LABELS))} END bucket,
               count(*) n, avg(p.won::INT) wr
        FROM players p JOIN matches m USING (match_id) WHERE {sc.where} GROUP BY ALL
    """)
    dur_w = dur.pivot(index="hero_id", columns="bucket", values="wr").add_prefix("wr_")
    df = df.merge(bans, on="hero_id", how="left").fillna({"bans": 0}).merge(dur_w, on="hero_id", how="left")
    df["win_rate"] = df.wins / df.games
    df["wr_lo"], df["wr_hi"] = wilson(df.wins, df.games)
    df["win_rate_shrunk"] = eb_shrink_rate(df.wins, df.games, 0.5)
    df["pick_rate"] = df.games / n_matches          # share of matches containing the hero
    df["ban_rate"] = df.bans / max(n_ban_matches, 1)
    df["matches_in_scope"] = n_matches
    df["low_sample"] = df.games < min_games
    df["scope"] = sc.name
    return df


def _design(m: pd.DataFrame, hero_idx: dict[int, int]):
    """Rows = matches from Team0's view, plus the mirrored Team1 view."""
    H = len(hero_idx)
    rows, cols, vals = [], [], []
    for r, (t0, t1) in enumerate(zip(m.heroes_team0, m.heroes_team1)):
        for h in t0:
            rows.append(r); cols.append(hero_idx[h]); vals.append(1.0)
        for h in t1:
            rows.append(r); cols.append(hero_idx[h]); vals.append(-1.0)
    n = len(m)
    Xh = sp.csr_matrix((vals, (rows, cols)), shape=(n, H))
    skill = m.skill_step_diff.fillna(0).to_numpy()[:, None]
    ranked = (m.match_mode == "Ranked").to_numpy(float)[:, None]
    side = np.ones((n, 1))
    X0 = sp.hstack([Xh, sp.csr_matrix(skill), sp.csr_matrix(side), sp.csr_matrix(side * ranked)]).tocsr()
    X1 = sp.hstack([-Xh, sp.csr_matrix(-skill), sp.csr_matrix(-side), sp.csr_matrix(-side * ranked)]).tocsr()
    y0 = (m.winning_team == "Team0").to_numpy(int)
    return sp.vstack([X0, X1]).tocsr(), np.concatenate([y0, 1 - y0]), n


def hero_strength(st: Store, sc: Scope, l2: float):
    m = st.q(f"""
        SELECT match_id, heroes_team0, heroes_team1, skill_step_diff, match_mode, winning_team
        FROM matches m WHERE {sc.where} AND len(heroes_team0) = 6 AND len(heroes_team1) = 6
    """)
    heroes = sorted({h for t in m.heroes_team0 for h in t} | {h for t in m.heroes_team1 for h in t})
    idx = {h: i for i, h in enumerate(heroes)}
    X, y, n = _design(m, idx)
    # C = 1/l2; hero coefficients are identified only up to a constant (each team has 6), so
    # the penalty also pins the mean hero coefficient near 0.
    model = LogisticRegression(C=1.0 / l2, fit_intercept=False, max_iter=500, solver="lbfgs")
    model.fit(X, y)
    beta = model.coef_[0]
    p = expit(X @ beta)
    W = p * (1 - p)
    # Fisher information; every match appears twice (mirrored), so halve it.
    info = 0.5 * (X.T.multiply(W) @ X).toarray() + l2 * np.eye(X.shape[1])
    cov = np.linalg.inv(info)
    H = len(heroes)
    # hero strengths are only identified relative to the mean hero: report b_h - mean(b),
    # whose covariance is C cov C^T with C the centering matrix (drops the unidentified
    # common shift, which otherwise dominates the marginal variance).
    Cm = np.eye(H) - 1.0 / H
    se_h = np.sqrt(np.clip(np.diag(Cm @ cov[:H, :H] @ Cm.T), 0, None))
    se = np.sqrt(np.diag(cov))
    b = beta[:H] - beta[:H].mean()
    res = pd.DataFrame(dict(hero_id=heroes, strength_logit=b, strength_se=se_h))
    # interpretable: win rate of a team with this hero instead of an average hero, all else equal
    res["adj_win_rate"] = expit(b)
    res["adj_wr_lo"], res["adj_wr_hi"] = expit(b - 1.96 * se_h), expit(b + 1.96 * se_h)
    extra_se = dict(skill_step_se=float(se[H]), side_team0_se=float(se[H + 1]))
    res["scope"] = sc.name
    extra = dict(skill_step_coef=float(beta[H]), side_team0_coef=float(beta[H + 1]),
                 side_ranked_extra=float(beta[H + 2]), n_matches=int(n), **extra_se)
    # expected Team0 win prob per match, for the residual analyses
    m["p_team0"] = p[:n]
    return res, extra, m[["match_id", "p_team0", "winning_team"]]


def pair_residuals(st: Store, sc: Scope, exp: pd.DataFrame, same_team: bool, min_games: int) -> pd.DataFrame:
    st.con.register("exp_df", exp)
    op = "=" if same_team else "<>"
    cond = "a.hero_id < b.hero_id" if same_team else "TRUE"
    df = st.q(f"""
        WITH e AS (
          SELECT p.match_id, p.hero_id, p.team, p.won::INT y,
                 CASE WHEN p.team = 'Team0' THEN x.p_team0 ELSE 1 - x.p_team0 END pe
          FROM players p JOIN exp_df x USING (match_id)
        )
        SELECT a.hero_id hero_a, b.hero_id hero_b, count(*) n, sum(a.y) wins,
               sum(a.y - a.pe) resid, sum(a.pe * (1 - a.pe)) var
        FROM e a JOIN e b ON a.match_id = b.match_id AND a.team {op} b.team AND a.hero_id <> b.hero_id AND {cond}
        GROUP BY ALL
    """)
    st.con.unregister("exp_df")
    df["win_rate"] = df.wins / df.n
    df["effect"] = df.resid / df.n            # win-rate points above expectation
    df["effect_se"] = np.sqrt(df["var"]) / df.n
    df["effect_shrunk"], df["effect_shrunk_sd"], tau = eb_shrink_effect(df.effect, df.effect_se)
    df["p_value"] = z_pvalue(df.effect, df.effect_se)
    df["q_value"] = bh_fdr(df.p_value)
    df["low_sample"] = df.n < min_games
    df["scope"] = sc.name
    return df.drop(columns=["resid", "var"])


def run(st: Store):
    s = st.s
    cfg = s["analysis"]["heroes"]
    g = GameData(s)
    names = g.heroes["name"].to_dict()
    out = {k: [] for k in ["hero_stats", "hero_strength", "hero_counters", "hero_synergy"]}
    meta = {}
    for sc in scopes(s):
        log.info("scope %s", sc.name)
        hs = hero_stats(st, sc, cfg["min_games"])
        if hs.games.sum() == 0:
            continue
        out["hero_stats"].append(hs)
        strength, extra, exp = hero_strength(st, sc, cfg["l2"])
        out["hero_strength"].append(strength)
        meta[sc.name] = extra
        out["hero_counters"].append(pair_residuals(st, sc, exp, False, cfg["min_pair_games"]))
        out["hero_synergy"].append(pair_residuals(st, sc, exp, True, cfg["min_pair_games"]))
    for k, parts in out.items():
        df = pd.concat(parts, ignore_index=True)
        for col, label in (("hero_id", "hero"), ("hero_a", "name_a"), ("hero_b", "name_b")):
            if col in df.columns:
                df[label] = df[col].map(names)
        st.write(k, df, meta if k == "hero_strength" else None)
