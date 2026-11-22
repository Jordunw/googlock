"""Incremental, resumable extraction of match data from the deadlock-api data dumps.

The `match_player` table is ~750 GB of parquet. We never download it whole: for every
remote file that can contain matches from the configured balance window we run a DuckDB
query over HTTP that only pulls the configured columns and rows (Ranked/Unranked, Normal
game mode, start_time inside the window) and write the result to a small local parquet.

Re-running is cheap: files already extracted (tracked in state.json by remote key) are
skipped. The dump is compacted hourly (delta -> residual -> base), so the same row can
appear in several remote files; `load.py` de-duplicates on (match_id, player_slot).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

from .config import Settings, load_settings

log = logging.getLogger("dla.ingest")


def _q(col: str) -> str:
    return '"' + col + '"'


def _alias(col: str) -> str:
    return col.replace(".", "_")


class Ingestor:
    def __init__(self, s: Settings):
        self.s = s
        self.cfg = s["ingest"]
        self.out_dir = s.data_dir / "raw" / self.cfg["table"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out_dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"files": {}}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- manifest
    def fetch_manifest(self) -> dict:
        req = urllib.request.Request(self.cfg["manifest_url"], headers={"User-Agent": "dla-analysis/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            m = json.load(r)
        (self.s.data_dir / "raw" / "manifest.json").write_text(json.dumps(m))
        return m

    def _con(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; SET TimeZone='UTC';")
        con.execute(f"SET threads={int(self.cfg.get('duckdb_threads_per_file', 4))}")
        con.execute(f"SET memory_limit='{self.cfg.get('memory_limit_per_file', '5GB')}'")
        con.execute("SET preserve_insertion_order=false")
        tmp = self.s.data_dir / "duckdb_tmp" / f"ingest_{threading.get_ident()}"
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")
        con.execute("SET http_retries=8; SET http_timeout=300000;")
        return con

    def _file_max_start(self, con, url: str) -> dt.datetime | None:
        r = con.execute(
            f"select max(stats_max) from parquet_metadata('{url}') where path_in_schema='start_time'"
        ).fetchone()[0]
        return dt.datetime.fromisoformat(r).astimezone(dt.timezone.utc) if r else None

    def first_partition(self, m: dict, start: dt.datetime) -> int:
        """Binary-search base partitions (match_id // 1e6 grows with time) for the first
        one whose data reaches `start`. Uses parquet footers only."""
        files = m["tables"][self.cfg["table"]]["files"]
        by_part: dict[int, list] = {}
        for f in files:
            if f["kind"] == "base":
                by_part.setdefault(f["partition"], []).append(f)
        parts = sorted(by_part)
        base = m["public_url"] + "/"
        con = self._con()

        def reaches(p):  # does partition p contain matches >= start?
            last = max(by_part[p], key=lambda f: f["key"])
            mx = self._file_max_start(con, base + last["key"])
            return mx is not None and mx >= start

        lo, hi = 0, len(parts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if reaches(parts[mid]):
                hi = mid
            else:
                lo = mid + 1
        # Files inside a partition are not time-ordered by key, so step back one partition
        # as a margin; the start_time filter discards anything older.
        p0 = parts[max(lo - 1, 0)]
        log.info("first partition reaching %s: %s (using %s)", start.isoformat(), parts[lo], p0)
        return p0

    def select_files(self, m: dict, p0: int) -> list[dict]:
        out = []
        for f in m["tables"][self.cfg["table"]]["files"]:
            if f["kind"] == "base":
                if f["partition"] >= p0:
                    out.append(f)
            else:  # delta / residual: keep if any rows land in partitions >= p0
                rbp = f.get("rows_by_partition") or {}
                if not rbp or any(int(p) >= p0 for p in rbp):
                    out.append(f)
        return out

    # -------------------------------------------------------------- extraction
    def _select_sql(self, url: str, start: dt.datetime, end: dt.datetime | None) -> str:
        cols = self.cfg["columns"]["match"] + self.cfg["columns"]["player"]
        sel = ", ".join(f"{_q(c)} AS {_alias(c)}" for c in cols)
        for name, expr in (self.cfg.get("derived") or {}).items():
            sel += f", ({expr}) AS {name}"
        mm = ", ".join(f"'{x}'" for x in self.cfg["match_modes"])
        gm = ", ".join(f"'{x}'" for x in self.cfg["game_modes"])
        where = [
            f"start_time >= TIMESTAMPTZ '{start.isoformat()}'",
            f"match_mode IN ({mm})",
            f"game_mode IN ({gm})",
        ]
        if end:
            where.append(f"start_time < TIMESTAMPTZ '{end.isoformat()}'")
        if self.cfg.get("require_team_win", True):
            where.append("match_outcome = 'TeamWin'")
        return f"SELECT {sel} FROM read_parquet('{url}') WHERE " + " AND ".join(where)

    def _signature(self) -> str:
        """Anything that changes extracted content invalidates previously extracted files."""
        import hashlib
        keys = ("columns", "derived", "match_modes", "game_modes", "require_team_win")
        payload = json.dumps({k: self.cfg.get(k) for k in keys} | {
            "start": self.s.patch_start.isoformat(),
            "end": self.s.patch_end.isoformat() if self.s.patch_end else None}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()

    def _local_name(self, key: str) -> Path:
        return self.out_dir / (key.split(f"{self.cfg['table']}/", 1)[1].replace("/", "__"))

    def extract(self, m: dict, f: dict, start, end) -> tuple[str, int, float]:
        url = m["public_url"] + "/" + f["key"]
        out = self._local_name(f["key"])
        tmp = out.with_suffix(".tmp")
        t = time.time()
        con = self._con()
        con.execute(
            f"COPY ({self._select_sql(url, start, end)}) TO '{tmp.as_posix()}' "
            "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 200000)"
        )
        n = con.execute(f"select count(*) from read_parquet('{tmp.as_posix()}')").fetchone()[0]
        con.close()
        tmp.replace(out)
        with self._lock:
            self.state["files"][f["key"]] = {
                "local": out.name, "rows": n, "kind": f["kind"], "built_at": f.get("built_at"),
                "window_start": start.isoformat(), "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            self.state_path.write_text(json.dumps(self.state, indent=1))
        return f["key"], n, time.time() - t

    def run(self, prune: bool = True, limit: int | None = None):
        start, end = self.s.patch_start, self.s.patch_end
        log.info("window: %s -> %s", start, end or "now")
        m = self.fetch_manifest()
        sig = self._signature()
        if self.state["files"] and self.state.get("signature") != sig:
            log.warning("window/columns/filters changed; re-extracting everything")
            for v in self.state["files"].values():
                (self.out_dir / v["local"]).unlink(missing_ok=True)
            self.state = {"files": {}}
        self.state["signature"] = sig
        self.state["window_start"] = start.isoformat()
        p0 = self.first_partition(m, start)
        todo_all = self.select_files(m, p0)
        todo = [f for f in todo_all if f["key"] not in self.state["files"]]
        todo.sort(key=lambda f: -f["bytes"])  # big ones first for better packing
        if limit:
            todo = todo[:limit]
        log.info("%d candidate files, %d new (%.1f GB remote)", len(todo_all), len(todo),
                 sum(f["bytes"] for f in todo) / 1e9)
        errors = []
        with ThreadPoolExecutor(max_workers=int(self.cfg.get("threads", 4))) as ex:
            futs = {ex.submit(self.extract, m, f, start, end): f for f in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                f = futs[fu]
                try:
                    key, n, secs = fu.result()
                    log.info("[%d/%d] %s rows=%d %.0fs", i, len(todo), key.rsplit("/", 1)[-1], n, secs)
                except Exception as e:  # keep going; failures are retried next run
                    errors.append((f["key"], repr(e)))
                    log.error("[%d/%d] FAILED %s: %s", i, len(todo), f["key"], e)
        if prune and not errors and not limit:
            live = {f["key"] for f in todo_all}
            for key in [k for k in self.state["files"] if k not in live]:
                p = self.out_dir / self.state["files"][key]["local"]
                p.unlink(missing_ok=True)
                del self.state["files"][key]
                log.info("pruned stale %s", key)
            self.state_path.write_text(json.dumps(self.state, indent=1))
        log.info("done; %d errors", len(errors))
        return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings")
    ap.add_argument("--limit", type=int, help="only extract N files (testing)")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--threads", type=int, help="override ingest.threads (e.g. 1 to retry OOM'd files)")
    ap.add_argument("--memory", help="override ingest.memory_limit_per_file, e.g. 14GB")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ing = Ingestor(load_settings(a.settings))
    if a.threads:
        ing.cfg["threads"] = a.threads
    if a.memory:
        ing.cfg["memory_limit_per_file"] = a.memory
    errs = ing.run(prune=not a.no_prune, limit=a.limit)
    raise SystemExit(1 if errs else 0)


if __name__ == "__main__":
    main()
