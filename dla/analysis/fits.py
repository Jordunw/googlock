"""Hero playstyle profiles and hero x item fit ("Crushing Fists on Paige").

hero_profiles      per hero: observed playstyle from matches (kill mix, engagement distances,
                   accuracy, headshots, parries, farm sources) + static stats/scaling from
                   deadlock-data (melee damage & scaling, weapon DPS, spirit scaling of kit).
hero_item_fit      per hero x item: hero-specific effect (from items.py), buy rate, and a
                   profile-predicted fit: for each item, ridge-regress the hero-specific
                   effects on standardized hero profiles across heroes (leave-one-hero-out),
                   so a hero gets credit if heroes *like it* benefit. Flags:
                     underused_proven     effect clearly positive and rarely bought
                     underused_predicted  profile predicts a strong fit, rarely bought,
                                          own data not contradicting
                   "Rarely bought" = buy rate below fits.rare_buy_rate (absolute) or in the
                   bottom fits.rare_buy_pct of heroes for that item (relative).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..gamedata import GameData
from .common import Store

log = logging.getLogger("dla.analysis.fits")

DIST_BINS = ["10", "20", "30", "40", "50", "75", "100", "Infinite"]
DIST_MID = {"10": 5, "20": 15, "30": 25, "40": 35, "50": 45, "75": 62.5, "100": 87.5, "Infinite": 120}


def _spirit_scaling(g: GameData, ability_key: str) -> float:
    """Sum of spirit-scaling coefficients found anywhere in an ability's data."""
    tot = 0.0

    def walk(v):
        nonlocal tot
        if isinstance(v, dict):
            sc = v.get("Scale")
            if isinstance(sc, dict) and sc.get("Type") == "spirit" and isinstance(sc.get("Value"), (int, float)):
                tot += float(sc["Value"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(g.raw_abilities.get(ability_key, {}))
    return tot


def hero_profiles(st: Store, g: GameData) -> pd.DataFrame:
    ob = " + ".join(f"coalesce(custom_stats['Outgoing Bullet Dist##{b}'], 0)" for b in DIST_BINS)
    oa = " + ".join(f"coalesce(custom_stats['Outgoing Ability Dist##{b}'], 0)" for b in DIST_BINS)
    obm = " + ".join(f"{DIST_MID[b]} * coalesce(custom_stats['Outgoing Bullet Dist##{b}'], 0)" for b in DIST_BINS)
    oam = " + ".join(f"{DIST_MID[b]} * coalesce(custom_stats['Outgoing Ability Dist##{b}'], 0)" for b in DIST_BINS)
    obc = " + ".join(f"coalesce(custom_stats['Outgoing Bullet Dist##{b}'], 0)" for b in ["10"])
    obs = st.q(f"""
        SELECT p.hero_id, count(*) games,
               sum(final_melee_kills) / sum(kills) melee_kill_share,
               sum(final_bullet_kills) / sum(kills) bullet_kill_share,
               sum(final_ability_kills) / sum(kills) ability_kill_share,
               sum(final_headshot_kills) / greatest(sum(final_bullet_kills), 1) headshot_kill_share,
               sum(final_shots_hit) / greatest(sum(final_shots_hit + final_shots_missed), 1) hit_rate,
               sum(final_hero_bullets_hit_crit) / greatest(sum(final_hero_bullets_hit), 1) crit_rate,
               sum({obm}) / greatest(sum({ob}), 1) bullet_engage_dist_m,
               sum({oam}) / greatest(sum({oa}), 1) ability_engage_dist_m,
               sum({obc}) / greatest(sum({ob}), 1) bullet_close_share,
               sum({oa}) / greatest(sum({ob}) + sum({oa}), 1) ability_damage_share,
               avg(coalesce(custom_stats['Parry Success'], 0)) parries_per_game,
               sum(final_gold_neutral_creep) / greatest(sum(net_worth), 1) jungle_soul_share,
               sum(final_gold_player) / greatest(sum(net_worth), 1) kill_soul_share,
               avg(final_self_healing / (m.duration_s / 60)) self_heal_per_min,
               avg(net_worth / (m.duration_s / 60)) souls_per_min
        FROM players p JOIN matches m USING (match_id) WHERE m.in_scope GROUP BY 1
    """)
    rows = []
    for hid, h in g.heroes.iterrows():
        raw = g.raw_heroes[h.key]
        ls = raw.get("LevelScaling") or {}
        w = raw.get("Weapon") or {}
        rows.append(dict(
            hero_id=hid, light_melee=raw.get("LightMeleeDamage"), heavy_melee=raw.get("HeavyMeleeDamage"),
            heavy_melee_scaling=ls.get("HeavyMeleeDamage"), bullet_dmg_scaling=ls.get("BulletDamage"),
            weapon_dps=w.get("DPS"), weapon_sustained_dps=w.get("SustainedDPS"), clip_size=w.get("ClipSize"),
            fire_rate=w.get("RoundsPerSecond"), bullet_damage=w.get("BulletDamage"),
            pellets=w.get("BulletsPerShot"), max_health=raw.get("MaxHealth"), move_speed=raw.get("MaxMoveSpeed"),
            kit_spirit_scaling=sum(_spirit_scaling(g, k) for k in h.abilities.values()),
        ))
    static = pd.DataFrame(rows)
    df = obs.merge(static, on="hero_id", how="left")
    df["hero"] = df.hero_id.map(g.heroes["name"])
    return df


def hero_item_fit(st: Store, prof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    he = pd.read_parquet(st.results / "hero_item_effects.parquet")
    feats = [c for c in prof.columns if c not in ("hero_id", "hero", "games")]
    P = prof.set_index("hero_id")[feats].astype(float)
    P = (P - P.mean()) / P.std().replace(0, 1)
    P = P.fillna(0)
    out = []
    for (scope, item), g in he.groupby(["scope", "item_id"]):
        g = g[g.hero_id.isin(P.index)].copy()
        if len(g) < cfg["min_heroes_for_fit"]:
            g["fit_predicted"] = np.nan
            out.append(g)
            continue
        X = P.loc[g.hero_id].to_numpy()
        y = g.hero_specific.to_numpy()
        w = 1 / (g.hero_specific_sd.to_numpy() ** 2 + 1e-6)
        pred = np.empty(len(g))
        for i in range(len(g)):   # leave-one-hero-out: a hero's prediction ignores its own data
            m = np.ones(len(g), bool)
            m[i] = False
            r = Ridge(alpha=cfg["ridge_alpha"]).fit(X[m], y[m], sample_weight=w[m])
            pred[i] = r.predict(X[i:i + 1])[0]
        g["fit_predicted"] = pred
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    # buy-rate percentile among heroes for this item (low = rarely bought on this hero)
    df["buy_rate_pct"] = df.groupby(["scope", "item_id"]).buy_rate.rank(pct=True)
    eff_lo = df.effect_shrunk - 1.64 * df.hero_specific_sd
    rare = (df.buy_rate < cfg.get("rare_buy_rate", 0.05)) | (df.buy_rate_pct <= cfg.get("rare_buy_pct", 1 / 3))
    df["underused_proven"] = (eff_lo > cfg["min_effect"]) & rare & ~df.low_sample
    df["underused_predicted"] = ((df.item_effect + df.fit_predicted) > cfg["min_effect"]) & \
        (df.fit_predicted > 0) & rare & (df.effect_shrunk > -cfg["min_effect"])
    return df


def run(st: Store):
    s = st.s
    g = GameData(s)
    prof = hero_profiles(st, g)
    st.write("hero_profiles", prof)
    st.write("hero_item_fit", hero_item_fit(st, prof, s["analysis"]["fits"]))
