"""Settings loader. All paths are resolved relative to the project root."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    raw: dict[str, Any]

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def deadlock_data(self) -> Path:
        return (ROOT / self.raw["paths"]["deadlock_data"]).resolve()

    @property
    def data_dir(self) -> Path:
        p = (ROOT / self.raw["paths"]["data_dir"]).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def patch_start(self) -> dt.datetime:
        v = self.raw["patch"]["patch_start"]
        if v == "auto":
            return auto_patch_start(self.deadlock_data)
        return _parse_ts(v)

    @property
    def patch_end(self) -> dt.datetime | None:
        v = self.raw["patch"].get("patch_end")
        return _parse_ts(v) if v else None


def _parse_ts(v) -> dt.datetime:
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def auto_patch_start(deadlock_data: Path) -> dt.datetime:
    """Latest non-hero-lab changelog date + 1 day (Valve posts late Pacific time)."""
    import json

    cfg = json.loads((deadlock_data / "changelogs" / "changelog_configs.json").read_text(encoding="utf-8"))
    dates = [v["date"] for v in cfg.values() if not v.get("is_hero_lab")]
    latest = max(dates)
    return _parse_ts(latest + "T00:00:00Z") + dt.timedelta(days=1)


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path) if path else ROOT / "settings.yaml"
    return Settings(yaml.safe_load(path.read_text(encoding="utf-8")))
