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
                     "sold_rate", "upgraded_rate", "heroes_n", "heroes_positive", "hero_mean_effect", "low_sample"],
    "hero_item_fit": ["scope", "hero_id", "hero", "item_id", "name", "tier", "n", "hero_specific", "effect_shrunk",
                      "buy_rate", "fit_predicted", "median_buy_min", "underused_proven", "underused_predicted",
                      "low_sample"],
    "hero_profiles": None,  # all columns
    # item_counters is assembled from item_vs_hero_win + item_vs_hero_kill (see _item_counters)
}


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
    return w


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
        data[name] = _columnar(df)
    ic = _item_counters(res)
    if ic is not None:
        data["item_counters"] = _columnar(ic)

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
    )
    html = TEMPLATE.read_text(encoding="utf-8")
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
