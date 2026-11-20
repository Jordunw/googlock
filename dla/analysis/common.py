"""Shared helpers: data access, analysis scopes, statistics."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from ..config import Settings

log = logging.getLogger("dla.analysis")


@dataclass(frozen=True)
class Scope:
    """A slice of matches every analysis is repeated over (bracket x ranked-only)."""
    name: str
    bracket: str | None      # None = all in-scope matches pooled
    ranked_only: bool

    @property
    def where(self) -> str:
        w = ["m.in_scope"]
        if self.bracket:
            w.append(f"m.bracket = '{self.bracket}'")
        if self.ranked_only:
            w.append("m.match_mode = 'Ranked'")
        return " AND ".join(w)


def scopes(s: Settings) -> list[Scope]:
    out = []
    for ranked in (False, True):
        suffix = " | Ranked" if ranked else ""
        out.append(Scope("All" + suffix, None, ranked))
        for b in s["skill"]["brackets"]:
            out.append(Scope(b + suffix, b, ranked))
    return out


class Store:
    """DuckDB views over data/clean, and writer for data/results."""

    def __init__(self, s: Settings):
        self.s = s
        self.clean = s.data_dir / "clean"
        self.results = s.data_dir / "results"
        self.results.mkdir(exist_ok=True)
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone='UTC'")
        self.con.execute(f"SET memory_limit='{s['clean'].get('memory_limit', '16GB')}'")
        self.con.execute(f"SET temp_directory='{(s.data_dir / 'duckdb_tmp').as_posix()}'")
        for t in ["players", "matches", "purchases", "abilities", "accounts", "deaths"]:
            p = self.clean / f"{t}.parquet"
            if p.exists():
                self.con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM '{p.as_posix()}'")
        # Deterministic match sample for the heavy purchase/death-level stages
        # (analysis.heavy_sample_fraction; 1.0 = everything). Hash-based so it is stable
        # across runs and keeps bracket proportions.
        frac = float(s["analysis"].get("heavy_sample_fraction", 1.0))
        self.sample_pred = "TRUE" if frac >= 1 else f"(hash(match_id) % 10000) < {int(frac * 10000)}"
        if (self.clean / "matches.parquet").exists():
            for t in ["players", "matches", "purchases", "deaths"]:
                if (self.clean / f"{t}.parquet").exists():
                    self.con.execute(f"CREATE OR REPLACE VIEW {t}_s AS SELECT * FROM {t} WHERE {self.sample_pred}")

    def q(self, sql: str, **kw) -> pd.DataFrame:
        return self.con.execute(sql, kw or None).df()

    def write(self, name: str, df: pd.DataFrame, meta: dict | None = None):
        df.to_parquet(self.results / f"{name}.parquet", index=False)
        if meta is not None:
            (self.results / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
        log.info("wrote %s (%d rows)", name, len(df))


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def eb_shrink_rate(k, n, prior_mean=None):
    """Empirical-Bayes beta-binomial shrinkage of rates k/n toward their pooled mean.
    Returns posterior means. Prior strength fitted by method of moments."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / np.maximum(n, 1)
    m = prior_mean if prior_mean is not None else k.sum() / max(n.sum(), 1)
    w = n / n.sum()
    var_obs = np.sum(w * (p - m) ** 2)
    var_samp = np.sum(w * m * (1 - m) / np.maximum(n, 1))
    var_true = max(var_obs - var_samp, 1e-6)
    a_plus_b = max(m * (1 - m) / var_true - 1, 1.0)
    return (k + m * a_plus_b) / (n + a_plus_b)


def eb_shrink_effect(est, se):
    """Normal-normal empirical Bayes: shrink effect estimates toward 0 by their noise.
    Returns (posterior mean, posterior sd, tau)."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    tau2 = max(np.nanmean(est[ok] ** 2) - np.nanmean(se[ok] ** 2), 1e-8) if ok.any() else 1e-8
    w = tau2 / (tau2 + se ** 2)
    return w * est, np.sqrt(w) * se, np.sqrt(tau2)


def bh_fdr(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order[:n], 1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def z_pvalue(est, se):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * stats.norm.sf(np.abs(np.asarray(est) / np.asarray(se)))
