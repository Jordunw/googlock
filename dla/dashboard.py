"""Build the self-contained HTML dashboard from data/results.

    python -m dla.dashboard            # writes data/dashboard/index.html

Data is embedded as columnar JSON so the file works offline and can be shared/published.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .analysis.common import scopes
from .config import load_settings
from .gamedata import GameData, slot_name

log = logging.getLogger("dla.dashboard")

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
IVJS = Path(__file__).with_name("itemvalue.js")       # browser item-value engine, inlined into the page

# columns shipped per results table (keeps the file small)
COLS = {
    "hero_stats": ["scope", "hero_id", "hero", "games", "win_rate", "wr_lo", "wr_hi", "win_rate_shrunk", "pick_rate",
                   "ban_rate", "kills", "deaths", "assists", "souls_per_min", "wr_<25m", "wr_25-35m", "wr_35-45m",
                   "wr_45m+", "low_sample"],
    "hero_strength": ["scope", "hero_id", "hero", "strength_logit", "strength_se", "adj_win_rate", "adj_wr_lo",
                      "adj_wr_hi"],
    "hero_counters": ["scope", "hero_a", "hero_b", "name_a", "name_b", "n", "win_rate", "effect_shrunk",
                      "effect_shrunk_sd", "q_value", "low_sample"],
    "hero_synergy": ["scope", "hero_a", "hero_b", "name_a", "name_b", "n", "win_rate", "effect_shrunk",
                     "effect_shrunk_sd", "q_value", "low_sample"],
    "item_effects": ["scope", "item_id", "name", "slot", "tier", "cost", "activation", "n", "raw_win_rate",
                     "expected_win_rate", "effect", "effect_se", "effect_per_1k_souls", "q_value", "median_buy_min",
                     "sold_rate", "upgraded_rate", "heroes_n", "heroes_positive", "hero_mean_effect", "core_heroes",
                     "low_sample"],
    "hero_item_fit": ["scope", "hero_id", "hero", "item_id", "name", "tier", "n", "hero_specific", "effect_shrunk",
                      "buy_rate", "fit_predicted", "median_buy_min", "underused_proven", "underused_predicted",
                      "core", "low_sample"],
    "hero_profiles": None,  # all columns
    # item_counters: item_vs_hero_win + item_vs_hero_kill + item_vs_hero_output (see _item_counters)
    # item_impact: item_output + item_state_split (see _item_impact); item value: computed in the page (itemvalue.js)
}
OUT_METRICS = ["dmg_out", "bullet_out", "ability_out", "dmg_taken", "bullet_in", "ability_in", "mitigated", "healing",
               "self_heal", "heal_prevented", "heal_lost", "kills", "deaths"]


def _item_impact(res: Path) -> pd.DataFrame | None:
    """Per (scope, item): output change after purchase (% of baseline) + win effect by game state."""
    po, ps = res / "item_output.parquet", res / "item_state_split.parquet"
    if not po.exists():
        return None
    o = pd.read_parquet(po)
    keep = ["scope", "item_id", "name", "slot", "tier", "n"]
    for m in OUT_METRICS:
        if f"{m}_pct" in o:
            o[f"{m}_pct_se"] = o[f"{m}_se"] / (o[m] / o[f"{m}_pct"]).abs()
            keep += [f"{m}_pct", f"{m}_pct_se"]
    o = o[keep]
    if ps.exists():
        s = pd.read_parquet(ps)
        w = s.pivot_table(index=["scope", "item_id"], columns="bucket",
                          values=["win_effect", "win_se", "win_n", "dmg_out_pct"], aggfunc="first")
        w.columns = [f"{v}_{b}" for v, b in w.columns]
        w = w.reset_index()
        if {"win_effect_ahead", "win_effect_behind"} <= set(w.columns):
            w["snowball"] = w.win_effect_ahead - w.win_effect_behind
            w["snowball_se"] = np.sqrt(w.win_se_ahead ** 2 + w.win_se_behind ** 2)
        o = o.merge(w, on=["scope", "item_id"], how="left")
    return o


def _item_value_inputs(res: Path) -> dict | None:
    """Inputs for the browser item-value engine (itemvalue.js computes everything live)."""
    p = res / "item_value_inputs.json"
    return json.loads(p.read_text()) if p.exists() else None


SHRED_KINDS = {"enemy_bullet_resist_down", "enemy_spirit_resist_down"}
DEBUFF_KINDS = SHRED_KINDS | {"slow", "stun_s", "silence_s", "disarm_s", "enemy_heal_down", "enemy_damage_down",
                              "enemy_spirit_damage_down", "enemy_fire_rate_down"}
BULLET_DEF = {"resist_bullet", "immune_bullet_s", "enemy_fire_rate_down", "reflect_bullet"}
SPIRIT_DEF = {"resist_spirit", "enemy_spirit_damage_down", "reflect_spirit"}


def _item_kinds(it: dict, rules: dict) -> set:
    """What an item card does, in item_rules.yaml kinds (property rules, remaps and added effects)."""
    props, ov = rules["props"], rules["items"].get(it["key"]) or {}
    remap, skip = ov.get("remap") or {}, set(ov.get("skip") or [])
    kinds = {e["kind"] for e in ov.get("add") or []}
    for b in it["blocks"]:
        for p in b["props"]:
            if p["k"] in skip:
                continue
            k = remap.get(p["k"]) or (props.get(p["k"]) or {}).get("kind")
            if k and k != "ignore" and p["v"]:
                kinds.add(k)
    return kinds


def _counterbuy_meta(s, iv_inputs: dict | None) -> dict:
    """Counterbuy planner defaults (settings.counterbuy) plus, per item id: team stacking (confirmed items
    and passive resist shred), refresh-only debuffs (actives) and the damage type it defends against."""
    cfg = dict(s.raw.get("counterbuy") or {})
    stacking, refresh, defense = {}, [], {}
    if iv_inputs:
        confirmed = cfg.get("team_stacking") or {}
        shred = float(cfg.get("shred_stacking_bonus", 0))
        for it in iv_inputs["items"]:
            kinds = _item_kinds(it, iv_inputs["rules"])
            iid = str(it["item_id"])
            active = str(it.get("activation") or "").lower() not in ("passive", "")
            if it["key"] in confirmed:
                stacking[iid] = float(confirmed[it["key"]])
            elif shred and kinds & SHRED_KINDS and not active:
                stacking[iid] = shred
            if active and kinds & DEBUFF_KINDS:
                refresh.append(iid)
            b, sp = bool(kinds & BULLET_DEF), bool(kinds & SPIRIT_DEF)
            if b or sp:
                defense[iid] = "both" if (b and sp) else "bullet" if b else "spirit"
        unknown = set(confirmed) - {it["key"] for it in iv_inputs["items"]}
        if unknown:
            log.warning("counterbuy.team_stacking: unknown item keys %s", sorted(unknown))
    cfg.update(team_stacking=stacking, refresh_only=refresh, defense=defense)
    return cfg


def _item_counters(res: Path) -> pd.DataFrame | None:
    """One row per (scope, item, enemy hero): win-delta signal + offensive / defensive kill odds."""
    pw, pk = res / "item_vs_hero_win.parquet", res / "item_vs_hero_kill.parquet"
    if not (pw.exists() and pk.exists()):
        return None
    keys = ["scope", "item_id", "enemy_hero"]
    w = pd.read_parquet(pw)[keys + ["item", "tier", "enemy", "n", "effect_with", "delta_shrunk", "delta_shrunk_sd",
                                    "q_value", "low_sample"]]
    w = w[w.n >= MIN_ROWS["item_vs_hero_win"][1]]
    k = pd.read_parquet(pk)
    k = k[k.holder_events >= MIN_ROWS["item_vs_hero_kill"][1]]
    for d, pre in (("offensive", "off"), ("defensive", "def")):
        x = k[k.direction == d][keys + ["holder_events", "log_or_shrunk", "log_or_shrunk_sd", "q_value", "low_sample"]]
        x = x.rename(columns={"holder_events": f"{pre}_n", "log_or_shrunk": f"{pre}_lor", "log_or_shrunk_sd": f"{pre}_lor_sd",
                              "q_value": f"{pre}_q", "low_sample": f"{pre}_low"})
        w = w.merge(x, on=keys, how="left")
    po = res / "item_vs_hero_output.parquet"
    if po.exists():
        o = pd.read_parquet(po)
        cols = {"n": "out_n", "e_dmg_out_pct": "e_dmg_pct", "e_healing_pct": "e_heal_pct",
                "dmg_taken_delta_pct": "taken_delta_pct"}
        if "e_dmg_out_se" in o:      # % SE from the absolute SE and the baseline implied by mean / pct
            o["e_dmg_pct_se"] = o.e_dmg_out_se / (o.e_dmg_out / o.e_dmg_out_pct).abs()
            cols["e_dmg_pct_se"] = "e_dmg_pct_se"
        o = o[keys + [c for c in cols if c in o]].rename(columns=cols)
        w = w.merge(o, on=keys, how="left")
    return w


def _hero_item_extras(res: Path, fit: pd.DataFrame) -> pd.DataFrame:
    """Add core-item landmark effects and per-hero output changes to hero_item_fit rows."""
    keys = ["scope", "hero_id", "item_id"]
    pc = res / "hero_core_items.parquet"
    if pc.exists():
        c = pd.read_parquet(pc)
        if not c.empty:
            c = c[keys + ["on_time_effect", "on_time_se", "on_time_rate", "landmark_s", "q_value"]] \
                .rename(columns={"q_value": "on_time_q"})
            fit = fit.merge(c, on=keys, how="left")
    ph = res / "hero_item_output.parquet"
    if ph.exists():
        h = pd.read_parquet(ph)
        cols = {f"{m}_pct": f"out_{m}_pct" for m in ("dmg_out", "dmg_taken", "healing") if f"{m}_pct" in h}
        fit = fit.merge(h[keys + list(cols)].rename(columns=cols), on=keys, how="left")
    return fit


# Rows below these sample sizes carry no usable signal and are not shipped (rows between
# these and the analysis min_* thresholds are shipped flagged low_sample).
MIN_ROWS = {"hero_counters": ("n", 100), "hero_synergy": ("n", 100), "hero_item_fit": ("n", 30),
            "item_vs_hero_win": ("n", 100), "item_vs_hero_kill": ("holder_events", 30)}


def _sig(v: float, digits: int = 4) -> float:
    return float(f"{v:.{digits}g}")


def _columnar(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            out[c] = [None if not np.isfinite(v) else _sig(v) for v in s]
        elif pd.api.types.is_bool_dtype(s):
            out[c] = [int(v) for v in s]
        elif pd.api.types.is_integer_dtype(s):
            out[c] = [int(v) for v in s]
        else:
            vals = [None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in s]
            uniq = sorted({v for v in vals if v is not None})
            if len(uniq) < len(vals) / 4:      # dictionary-encode repetitive strings
                pos = {v: i for i, v in enumerate(uniq)}
                out[c] = dict(d=uniq, i=[-1 if v is None else pos[v] for v in vals])
            else:
                out[c] = vals
    return out


def build(settings_path=None) -> Path:
    s = load_settings(settings_path)
    res = s.data_dir / "results"
    g = GameData(s)
    data = {}
    for name, cols in COLS.items():
        p = res / f"{name}.parquet"
        if not p.exists():
            log.warning("missing %s; tab will be empty", name)
            continue
        df = pd.read_parquet(p)
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        if "slot" in df.columns:          # results written before the slot rename
            df["slot"] = df["slot"].map(slot_name)
        if name in MIN_ROWS:
            col, lo = MIN_ROWS[name]
            df = df[df[col] >= lo]
        if name == "hero_item_fit":
            df = _hero_item_extras(res, df)
        data[name] = _columnar(df)
    ic = _item_counters(res)
    if ic is not None:
        data["item_counters"] = _columnar(ic)
    ii = _item_impact(res)
    if ii is not None:
        data["item_impact"] = _columnar(ii)
    iv_inputs = _item_value_inputs(res)

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    clean = (s.data_dir / "clean").as_posix()
    win = con.execute(f"""
        SELECT strftime(min(start_time), '%Y-%m-%d %H:%M UTC'), strftime(max(start_time), '%Y-%m-%d %H:%M UTC'), count(*), count(*) FILTER (WHERE match_mode = 'Ranked'),
               count(*) FILTER (WHERE in_scope)
        FROM '{clean}/matches.parquet'""").fetchone()
    by_scope = {}
    for sc in scopes(s):
        w = sc.where.replace("m.", "")
        by_scope[sc.name] = con.execute(f"SELECT count(*) FROM '{clean}/matches.parquet' WHERE {w}").fetchone()[0]
    meta = dict(
        generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        patch_start=s.patch_start.strftime("%Y-%m-%d %H:%M UTC"),
        first_match=str(win[0])[:16], last_match=str(win[1])[:16], matches=win[2], ranked=win[3], in_scope=win[4],
        matches_by_scope=by_scope, game_version=g.version.get("VersionDate", "?"),
        brackets=list(s["skill"]["brackets"]), focus_bracket=s["skill"].get("focus_bracket"),
        min_effect=s["analysis"]["fits"]["min_effect"],
        min_known_players=s["skill"]["min_known_players"],
        season_starts=[str(x) for x in s["skill"].get("season_starts") or []],
        core_buy_rate=s["analysis"]["items"].get("core_buy_rate", 0.5),
        item_value=iv_inputs,
        counterbuy=_counterbuy_meta(s, iv_inputs),
    )
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__ITEMVALUE_JS__*/", IVJS.read_text(encoding="utf-8"))
    payload = json.dumps(dict(meta=meta, data=data), separators=(",", ":"))
    html = html.replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    out = s.data_dir / "dashboard"
    out.mkdir(exist_ok=True)
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    log.info("dashboard written to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(a.settings)


if __name__ == "__main__":
    main()
