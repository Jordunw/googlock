"""Static game data from the deadlock-data repo, keyed by the numeric ids used in match data.

Match dumps reference heroes by `Id` and items/abilities by MurmurHash2(class_name,
seed=0x31415926). We rebuild those ids locally so no external lookup is needed.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd

from .config import Settings


# Shop categories as named in game (the game files still use the old internal names)
SLOT_NAMES = {"Tech": "Spirit", "Armor": "Health"}


def slot_name(raw):
    return SLOT_NAMES.get(raw, raw)


def murmur2(key: str, seed: int = 0x31415926) -> int:
    data = key.encode()
    m, r = 0x5BD1E995, 24
    n = len(data)
    h = (seed ^ n) & 0xFFFFFFFF
    i = 0
    while n >= 4:
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * m) & 0xFFFFFFFF
        k ^= k >> r
        k = (k * m) & 0xFFFFFFFF
        h = ((h * m) & 0xFFFFFFFF) ^ k
        i += 4
        n -= 4
    if n == 3:
        h ^= data[i + 2] << 16
    if n >= 2:
        h ^= data[i + 1] << 8
    if n >= 1:
        h ^= data[i]
        h = (h * m) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * m) & 0xFFFFFFFF
    h ^= h >> 15
    return h


class GameData:
    def __init__(self, s: Settings):
        self.s = s
        self.root: Path = s.deadlock_data
        self.rules = s["gamedata"]

    def _json(self, name):
        return json.loads((self.root / "json" / name).read_text(encoding="utf-8"))

    @property
    def version(self) -> dict:
        lines = (self.root / "version.txt").read_text().splitlines()
        return dict(l.split("=", 1) for l in lines if "=" in l)

    @cached_property
    def raw_items(self) -> dict:
        return self._json("item-data.json")

    @cached_property
    def raw_heroes(self) -> dict:
        return self._json("hero-data.json")

    @cached_property
    def raw_abilities(self) -> dict:
        return self._json("ability-data.json")

    @cached_property
    def items(self) -> pd.DataFrame:
        """Shop items (upgrade_*). `valid` = buyable in ranked/unranked per the rules."""
        rows = []
        for key, v in self.raw_items.items():
            if not v.get("Name") or v.get("Cost") is None:
                continue
            sb = bool(v.get("StreetBrawl")) or v.get("Cost") == self.rules["street_brawl_price"]
            dis = bool(v.get("IsDisabled"))
            rows.append(dict(
                item_id=murmur2(key), key=key, name=v["Name"], slot=slot_name(v.get("Slot")), tier=v.get("Tier"),
                cost=v.get("Cost"), activation=v.get("Activation"), is_imbue=bool(v.get("IsImbue")),
                components=v.get("Components") or [], street_brawl=sb, disabled=dis,
                valid=not sb and not (dis and self.rules["exclude_disabled_items"]),
            ))
        df = pd.DataFrame(rows)
        key2id = dict(zip(df.key, df.item_id))
        df["component_ids"] = df.components.map(lambda cs: [key2id[c] for c in cs if c in key2id])
        return df.set_index("item_id")

    @cached_property
    def heroes(self) -> pd.DataFrame:
        rows = []
        for key, v in self.raw_heroes.items():
            if v.get("Id") is None:
                continue
            rows.append(dict(
                hero_id=v["Id"], key=key, name=v.get("Name"), type=v.get("Type"),
                in_development=bool(v.get("InDevelopment")), disabled=bool(v.get("IsDisabled")),
                max_health=v.get("MaxHealth"), move_speed=v.get("MaxMoveSpeed"),
                dps=(v.get("Weapon") or {}).get("DPS"), sustained_dps=(v.get("Weapon") or {}).get("SustainedDPS"),
                abilities={int(k): a["Key"] for k, a in (v.get("BoundAbilities") or {}).items()},
            ))
        return pd.DataFrame(rows).set_index("hero_id")

    @cached_property
    def abilities(self) -> pd.DataFrame:
        owner = {}
        for hid, h in self.heroes.iterrows():
            for slot, k in h.abilities.items():
                owner[k] = (hid, slot)
        rows = [dict(ability_id=murmur2(k), key=k, name=v.get("Name"),
                     hero_id=owner.get(k, (None, None))[0], slot=owner.get(k, (None, None))[1])
                for k, v in self.raw_abilities.items()]
        return pd.DataFrame(rows).set_index("ability_id")

    @cached_property
    def id_names(self) -> dict[int, str]:
        """Any numeric id seen in match `items_item_id` -> human name/key."""
        d = {i: f"{r.name}" for i, r in self.abilities.iterrows()}
        d.update({i: r["name"] for i, r in self.items.iterrows()})
        return d

    def hero_name(self, hero_id: int) -> str:
        return self.heroes["name"].get(hero_id, str(hero_id))
