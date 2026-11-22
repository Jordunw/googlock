"""Item value from game data alone: what an item's numbers are worth, per soul.

    python -m dla.itemvalue            # writes data/results/item_value*.parquet + item_value_inputs.json
    python -m dla.itemvalue --check    # also checks the browser engine (itemvalue.js) gives the same numbers

No match data or win rates are used. For a reference hero at a few game stages (souls ->
boons from the soul-unlock table; HP, weapon DPS and spirit scaling from hero data -- the
average over all heroes, or one chosen hero; assumed item stats and resists from
settings.item_value.stages) every property on every item card is turned into a small set
of *channels*, measured over one reference fight of `combat_s` seconds (one fight per
minute of play):

    ehp            effective HP in the fight (HP, resists, barriers, immunities, enemy damage cuts)
    heal           HP healed during the fight (regen, procs, lifesteal); priced like ehp
    recovery       HP regained per minute between fights (out-of-combat regen and heals)
    weapon/spirit/melee   damage dealt in the fight (incl. procs and enemy resist shred)
    heal_denied    enemy healing prevented in the fight
    slow, hard_cc  slow-%-seconds and stun-equivalent seconds per fight; priced from the
                   damage a disabled enemy does not deal (stun_dps_multiple x incoming DPS)
    clip, bullet_speed, move, sprint, stamina, range, charges, status_resist, souls

How often a property is up follows the card: Innate = always; Passive/Active blocks with a
cooldown fire min(60/cd, max(1, combat_s/cd)) times per fight, capped by how often their
trigger happens (item_rules.yaml `trigger`: melee hits, casts, headshots), and buffs last
their duration; "ConditionallyApplied" values without a cooldown are up `cond_uptime` of
the fight. Mechanics the card does not express as numbers are added in item_rules.yaml.

Channels are priced in souls from the shop's own single-stat items (item_value.calibrate,
solved in order: Extra Health prices ehp, Rapid Rounds weapon damage, Extra Spirit spirit
damage, Extra Regen recovery, ...), so ROI = value / cost reads "worth this many souls of
plain stat items" and, for the average hero, the reference items are exactly 1.0.

For one hero, quantities come from that hero (its HP, gun, spirit scaling of its kit) and
are priced at the *average hero's* rates by default, so Extra Spirit is worth more than
1.0 on a hero whose abilities scale hard with spirit power and less on a gun hero
(pricing="own" calibrates on the hero itself instead: a within-hero ranking).

The arithmetic is mirrored in itemvalue.js, which the dashboard runs live so the hero and
the assumptions can be changed in the browser. Keep the two in step; --check compares them.

Outputs: item_value (average hero, item x stage: channel quantities and souls, value, roi,
unmodeled properties), item_value_channels (long format), item_value.meta.json (reference
heroes, prices, assumptions), item_value_inputs.json (everything the browser engine needs).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Settings, load_settings
from .gamedata import GameData

log = logging.getLogger("dla.itemvalue")

RULES = Path(__file__).with_name("item_rules.yaml")
JS = Path(__file__).with_name("itemvalue.js")
CHANNELS = ["ehp", "heal", "recovery", "weapon", "spirit", "melee", "heal_denied", "slow", "hard_cc", "clip",
            "bullet_speed", "move", "sprint", "stamina", "range", "charges", "status_resist", "souls", "items"]
CHANNEL_UNITS = {"ehp": "effective HP per fight", "heal": "HP healed per fight",
                 "recovery": "HP regained per min out of combat", "weapon": "weapon dmg per fight",
                 "spirit": "spirit dmg per fight", "melee": "melee dmg per fight",
                 "heal_denied": "enemy healing denied per fight", "slow": "slow %-s per fight",
                 "hard_cc": "stun-equivalent s per fight", "clip": "% clip size", "bullet_speed": "% bullet speed",
                 "move": "m/s (fight uptime)", "sprint": "m/s sprint", "stamina": "stamina", "range": "% ability range",
                 "charges": "ability charges", "status_resist": "% status resist", "souls": "souls gained",
                 "items": "souls of cooldown-item value gained"}
DURATION_KEYS = {"AbilityDuration", "BuffDuration", "DebuffDuration", "SlowDuration", "RegenDuration",
                 "RegenerationDuration", "BarrierDuration", "CasterBuffDuration", "MovementSpeedBonusDuration",
                 "FireRateSlowDuration", "InvisDuration", "DotDuration", "WeaponDamageBonusDuration",
                 "BonusShotsDuration", "AmbushDuration"}
# kinds that happen once per activation (value x uses) rather than being "up" a share of the fight
INSTANT = {"heal", "heal_pct_hp", "recovery_heal", "barrier", "spirit_dmg", "spirit_pct_hp", "bullet_proc",
           "bullet_proc_pct", "stun_s", "silence_s", "disarm_s", "hp_pct_full", "immune_bullet_s", "immune_all_s",
           "spirit_amp_proc"}
OVER_DURATION = {"spirit_dps", "spirit_pct_hp_dps", "slow", "reflect_bullet", "reflect_spirit"}
HERO_FIELDS = ["max_health", "ls_hp", "ls_sp", "ls_dps", "ls_bullet", "dps", "bullet", "bps", "rps", "clip", "reload",
               "melee", "light", "heavy", "ls_light", "ls_heavy"]
ABILITY_RULES = Path(__file__).with_name("ability_rules.yaml")
# generic reading of ability data (ability_rules.yaml describes the exceptions)
ABIL_DUR = ["AbilityDuration", "DotDuration", "BurnDuration", "VenomDuration", "DebuffDuration", "AuraDuration",
            "ChannelDuration", "TotalDuration", "Duration", "AbilityChannelTime"]
DMG_KEY = re.compile(r"damage|dps", re.I)
DMG_EXCL = re.compile(r"reduc|resist|taken|absorb|mitigat|heal|amp|pct|percent|cap|boss|scale|radius|range|permeter"
                      r"|stolen|attack|todamage|perlevel|falloff|minion|trooper|npc|self|bonusdamagepct|multiplier", re.I)
HEAL_KEY = re.compile(r"heal|hps|regen|lifedrain", re.I)
HEAL_EXCL = re.compile(r"pct|percent|kill|pickup|amp|reduc|prevent|lifesteal|onhit|shared|external|delay|duration|"
                       r"cooldown|radius|range|penalty|bonusmax", re.I)
PER_S = re.compile(r"dps|persecond|hps|regen|lifedrain", re.I)
PER_HIT = re.compile(r"perbullet|pershot|onhit", re.I)


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"^\s*(-?[\d.]+)", v)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------- inputs (game data -> plain JSON)
def _vs(v) -> tuple:
    """A property value -> (base, spirit coef, weapon-damage coef, light-melee coef, heavy-melee coef).
    Weapon coefficients scale the value with the hero's weapon damage bonus (x (1 + coef x bonus));
    melee coefficients add coef x the hero's melee damage."""
    if isinstance(v, dict):
        b = _num(v.get("Value"))
        c = {"spirit": 0.0, "weapon": 0.0, "melee": 0.0, "heavy_melee": 0.0}
        for x in (v.get("Scale") if isinstance(v.get("Scale"), list) else [v.get("Scale")]):
            if not isinstance(x, dict) or not isinstance(x.get("Value"), (int, float)):
                continue
            t, val = x.get("Type"), float(x["Value"])
            if t == "spirit":
                c["spirit"] += val
            elif t in ("weapon_damage_increase", "weapon_damage"):
                c["weapon"] += val
            elif t == "weapon_power":                       # given in percent
                c["weapon"] += val / 100
            elif t in ("melee", "heavy_melee"):
                c[t] += val
        return b, c["spirit"], c["weapon"], c["melee"], c["heavy_melee"]
    return _num(v), 0.0, 0.0, 0.0, 0.0


def _expr(e, env: dict) -> float:
    """A number, or an arithmetic expression over ability properties / vars (ability_rules.yaml)."""
    if e is None or isinstance(e, (int, float)):
        return e
    return float(eval(str(e), {"__builtins__": {}}, {**env, "min": min, "max": max}))  # trusted local rules file


def digest_ability(key: str, raw: dict, slot: int, rule: dict | None, desc: str = "",
                   defaults: dict | None = None) -> dict:
    """One ability -> cooldown, damage / healing terms and CC per cast (see ability_rules.yaml).
    desc: the localized description (raw), used for CC keywords the data has no duration for."""
    rule = rule or {}
    props = {k: _vs(v) for k, v in raw.items() if k != "Upgrades"}
    props = {k: v for k, v in props.items() if v[0] is not None}
    env = {k: v[0] for k, v in props.items()} | (rule.get("vars") or {})
    cd = props.get("AbilityCooldown", (None,))[0]
    charges = props.get("AbilityCharges", (1,))[0] or 1
    dur = next((props[k][0] for k in ABIL_DUR if props.get(k, (0,))[0]), None)
    ZERO = (0.0, 0.0, 0.0, 0.0, 0.0)
    aoe = any("Radius" in k and "Spawn" not in k for k in raw)
    bits = raw.get("BehaviourBits") or []
    if cd and dur and any("Toggle" in b and "Crouch" not in b for b in bits):
        cd += dur                           # toggles: the cooldown starts when switched off
    uses = _expr(rule.get("uses"), env)
    passive = not cd or cd <= 0 or uses == 0

    def term(t: dict) -> dict:
        if "blend" in t:
            lo, hi, wgt = t["blend"]
            v0, v1 = props.get(lo, ZERO), props.get(hi, ZERO)
            vals = [(1 - wgt) * (a or 0) + wgt * (b or 0) for a, b in zip(v0, v1)]
        elif "value" in t:                  # a fixed amount (x `of`), no scaling of its own
            vals = [_expr(t["value"], env), 0.0, 0.0, 0.0, 0.0]
        else:
            vals = list(props.get(t["key"], ZERO))
        base, coef, wcoef, mcoef, hcoef = [float(x or 0) for x in vals]
        if "coef" in t:                      # explicit spirit scaling (e.g. calibrated from play)
            coef = float(_expr(t["coef"], env))
        per = t.get("per", "cast")
        mult = _expr(t.get("mult"), env)
        if mult is None:
            mult = dur if per == "second" else 1.0      # None for "second" = default_dot_s at run time
        cdv = props.get(t["cd_key"], (None,))[0] if t.get("cd_key") else _expr(t.get("cd"), env)
        chan = t.get("chan") or ("weapon" if wcoef else "melee" if (mcoef or hcoef) else "spirit")
        return dict(base=base, coef=coef, wcoef=wcoef, mcoef=mcoef, hcoef=hcoef, chan=chan, per=per, mult=mult,
                    window=_expr(t.get("window"), env) if "window" in t else (None if passive else dur),
                    cd=cdv, hits_needed=_expr(t.get("hits_needed"), env) or 1.0, on=t.get("on", "hit"),
                    of=t.get("of"), aoe=bool(t.get("aoe", aoe and per in ("cast", "second"))))

    def gun_term(t: dict) -> dict:          # a window of boosted shooting (fire rate / damage %)
        w = t.get("window", "AbilityDuration" if "AbilityDuration" in env else dur)
        return dict(window=_expr(w, env) if w is not None else None,
                    fire_pct=_expr(t.get("fire_pct", 0), env) or 0.0, dmg_pct=_expr(t.get("dmg_pct", 0), env) or 0.0,
                    hit=_expr(t.get("hit"), env), aoe=bool(t.get("aoe", False)))

    if "terms" in rule:
        terms = [term(t) for t in rule["terms"]]
    else:                                   # generic reading of the damage properties
        dmg = [k for k, v in props.items() if DMG_KEY.search(k) and not DMG_EXCL.search(k) and any(v)]
        terms, used = [], set()
        for k in dmg:                       # Min*/Max* pairs -> 50/50 blend
            if "Min" in k and k.replace("Min", "Max") in dmg:
                per = "second" if PER_S.search(k) else "hit" if PER_HIT.search(k) else "cast"
                terms.append(term({"blend": [k, k.replace("Min", "Max"), 0.5], "per": per}))
                used |= {k, k.replace("Min", "Max")}
        rest = [k for k in dmg if k not in used]
        for k in rest:
            if PER_S.search(k):
                terms.append(term({"key": k, "per": "second"}))
            elif PER_HIT.search(k):
                terms.append(term({"key": k, "per": "hit"}))
        inst = [k for k in rest if not PER_S.search(k) and not PER_HIT.search(k)]
        chan_of = lambda k: "weapon" if props[k][2] else "melee" if (props[k][3] or props[k][4]) else "spirit"
        for ch in ("spirit", "weapon", "melee"):   # instant values of one kind are usually alternatives:
            ks = [k for k in inst if chan_of(k) == ch]   # keep the largest of each (spirit, weapon, melee)
            if ks:
                terms.append(term({"key": max(ks, key=lambda k: (sum(props[k][1:]), props[k][0] or 0))}))
    if "gun" in rule:
        gun_terms = [gun_term(t) for t in rule["gun"]]
    else:                                   # generic: fire-rate / bullet-damage buffs for a duration
        fr, amp = props.get("BonusFireRate", ZERO)[0] or 0.0, props.get("BulletDamageAmp", ZERO)[0] or 0.0
        gun_terms = [gun_term({"fire_pct": fr, "dmg_pct": amp})] if (fr > 0 or amp > 0) and dur and not passive else []
    if "heal" in rule:
        heal = [term(t) for t in rule["heal"]]
    else:
        hk = [k for k, v in props.items() if HEAL_KEY.search(k) and not HEAL_EXCL.search(k) and any(v[:2])]
        heal = [term({"key": k, "per": "second" if PER_S.search(k) else "cast"}) for k in hk if PER_S.search(k)]
        inst = [k for k in hk if not PER_S.search(k)]
        if inst:
            heal.append(term({"key": max(inst, key=lambda k: (props[k][1], props[k][0] or 0))}))
        for h in heal:
            h["aoe"] = False
    for h in heal:
        h["chan"] = "heal"
    if "cc" in rule:
        cc = {k: float(_expr(v, env)) for k, v in rule["cc"].items()}
    else:
        g = lambda *ks: next((props[k][0] for k in ks if props.get(k, (0,))[0]), 0.0)
        mods = " ".join(str(v.get("Class", "")) for v in raw.values() if isinstance(v, dict))
        cc = dict(stun=g("StunDuration") + g("SleepDuration"), root=g("ImmobilizeDuration"),
                  silence=g("SilenceDuration", "DebuffDuration") if "Silence" in mods else 0.0,
                  disarm=g("DisarmDuration", "DebuffDuration") if "Disarm" in mods else 0.0,
                  slow=abs(g("SlowPercent")) * (g("SlowDuration") or dur or 0.0))
        # displacement the data has no duration for, from the description's keywords
        if not cc["stun"]:
            d = defaults or {}
            if "'Stun'" in desc:
                cc["stun"] = float(d.get("stun_s", 1.0))
            elif "'KnockUp'" in desc:
                cc["stun"] = float(d.get("knockup_s", 0.6))
            elif "'KnockBack'" in desc or "'Pull" in desc:
                cc["stun"] = float(d.get("displace_s", 0.4))
        if not cc["root"] and "'Immobilize'" in desc:
            cc["root"] = float((defaults or {}).get("immobilize_s", 1.0))
    return dict(key=key, name=str(raw.get("Name") or key), slot=int(slot), ult=int(slot) == 4, cd=cd,
                charges=float(charges), uses=uses, passive=bool(passive), targets=_expr(rule.get("targets"), env),
                dur=dur, aoe=aoe or rule.get("targets") is not None, terms=terms, gun=gun_terms, heal=heal,
                refresh=bool(rule.get("refresh", False)), copy_ult=_expr(rule.get("copy_ult"), env),
                amp=[dict(kind=a.get("kind", "damage"), pct=float(_expr(a["pct"], env)) * float(_expr(a.get("stacks", 1), env)),
                          window=_expr(a.get("window"), env), targets=float(_expr(a.get("targets", 1), env)))
                     for a in rule.get("amp") or []],
                cc={k: v for k, v in cc.items() if v}, src="rule" if rule else "generic", note=rule.get("note"))


def hero_table(g: GameData, A: dict) -> tuple[list[dict], dict]:
    """Per released hero: the stats the engine uses (missing values filled with the average)."""
    doc = yaml.safe_load(ABILITY_RULES.read_text(encoding="utf-8")) or {}
    rules, defaults = doc.get("abilities") or {}, doc.get("defaults") or {}
    loc_path = g.root / "localizations" / "english.json"
    loc = json.loads(loc_path.read_text(encoding="utf-8")) if loc_path.exists() else {}
    out, seen = [], set()
    for h in g.raw_heroes.values():
        hid = h.get("Id")
        if not hid or h.get("InDevelopment") or h.get("IsDisabled") or hid not in g.heroes.index:
            continue
        name, abil = g.heroes.loc[hid, "name"], g.heroes.loc[hid, "abilities"] or {}
        if "dummy" in str(name).lower() or len(abil) < 4:
            continue
        ls, w = h.get("LevelScaling") or {}, h.get("Weapon") or {}
        light, heavy = _num(h.get("LightMeleeDamage")), _num(h.get("HeavyMeleeDamage"))
        abilities = [digest_ability(k, g.raw_abilities.get(k, {}), s, rules.get(k), str(loc.get(k + "_desc", "")),
                                    defaults) for s, k in sorted(abil.items())]
        seen |= set(abil.values())
        out.append(dict(id=int(hid), name=str(name), max_health=_num(h.get("MaxHealth")), ls_hp=_num(ls.get("MaxHealth")),
                        ls_sp=_num(ls.get("TechPower")), ls_dps=_num(ls.get("DPS")), ls_bullet=_num(ls.get("BulletDamage")),
                        dps=_num(w.get("DPS")), bullet=_num(w.get("BulletDamage")), bps=_num(w.get("BulletsPerShot")),
                        rps=_num(w.get("RoundsPerSecond")), clip=_num(w.get("ClipSize")),
                        ammo=_num(w.get("AmmoConsumedPerShot")) or 1.0, reload=_num(w.get("ReloadTime")),
                        clip_sp=_num((h.get("SpiritScaling") or {}).get("ClipSize")) or 0.0,
                        melee=(light + heavy) if light is not None and heavy is not None else None,
                        light=light, heavy=heavy, ls_light=_num(ls.get("LightMeleeDamage")),
                        ls_heavy=_num(ls.get("HeavyMeleeDamage")), abilities=abilities))
    unknown = set(rules) - seen
    if unknown:
        log.warning("ability_rules.yaml entries for unknown abilities: %s", sorted(unknown))
    avg = {f: float(np.mean([h[f] for h in out if h[f] is not None])) for f in HERO_FIELDS}
    for h in out:
        for f in HERO_FIELDS:
            if h[f] is None:
                h[f] = avg[f]
    avg.update(id=0, name="Average hero", ammo=1.0, clip_sp=0.0, abilities=None)
    return sorted(out, key=lambda h: h["name"]), avg


def digest_items(g: GameData) -> list[dict]:
    """Item cards reduced to what the engine reads: per block its type, cooldown, duration, proc
    chance, stacks and properties (value, spirit / boon scaling, conditional flag)."""
    cards = json.loads((g.root / "json" / "item-cards.json").read_text(encoding="utf-8"))
    items = g.items[g.items.valid].reset_index()
    out = []
    for it in items.itertuples():
        card = cards.get(it.key) or {}
        other = {k: v.get("Value") for k, v in (card.get("Other") or {}).items()}
        item_cd = _num(other.get("AbilityCooldown")) or _num((g.raw_items.get(it.key) or {}).get("AbilityCooldown"))
        blocks = []
        for bk in [b for b in card if b.startswith("Info")]:
            b = card[bk]
            btype = b.get("Type")
            props = (b.get("Main") or []) + (b.get("Alt") or [])
            pval = {p["Key"]: _num(p.get("Value")) for p in props}
            cd = _num(b.get("Cooldown")) or _num(b.get("ChargeUp"))
            if not cd and btype == "Active":
                cd = item_cd
            durs = [pval[k] for k in pval if k in DURATION_KEYS and pval[k]]
            dur = max(durs) if durs else None
            if not dur and (cd or btype == "Active"):
                dur = _num(other.get("AbilityDuration")) or _num(other.get("BuffDuration"))
            pc = pval.get("ProcChance") or _num(other.get("ProcChance"))
            stacks = pval.get("MaxStacks") or pval.get("MaxArmorStacks") or _num(other.get("MaxStacks")) or 1
            P = []
            for p in props:
                sc, st_, sv = p.get("Scale"), None, None
                if isinstance(sc, dict) and sc.get("Type") in ("spirit", "power_increase") \
                        and isinstance(sc.get("Value"), (int, float)) and not isinstance(sc.get("Value"), bool):
                    st_, sv = sc["Type"], float(sc["Value"])
                P.append(dict(k=p["Key"], v=_num(p.get("Value")), st=st_, sv=sv,
                              c=p.get("UsageFlags") in ("ConditionallyApplied", "ConditionallyEnemyApplied")))
            blocks.append(dict(t=btype, cd=cd or None, dur=dur or None, pc=pc or None, stacks=stacks,
                               pval={k: v for k, v in pval.items() if v is not None}, props=P))
        out.append(dict(item_id=int(it.item_id), key=it.key, name=it.name, slot=it.slot, tier=int(it.tier),
                        cost=float(it.cost), activation=it.activation, item_cd=item_cd, blocks=blocks))
    return out


def build_inputs(s: Settings) -> dict:
    g = GameData(s)
    A = dict(s["item_value"])
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    soul = json.loads((g.root / "json" / "soul-unlock-data.json").read_text(encoding="utf-8"))
    heroes, avg = hero_table(g, A)
    items = digest_items(g)
    unknown = set(rules.get("items") or {}) - {i["key"] for i in items}
    if unknown:
        log.warning("item_rules.yaml overrides for unknown item keys: %s", sorted(unknown))
    return dict(
        A=A, heroes=heroes, avg_hero=avg, items=items,
        boon_souls=sorted(float(x["RequiredSouls"]) for x in soul if x.get("PowerIncrease")),
        rules=dict(props={k: (v if isinstance(v, dict) else {"kind": v}) for k, v in rules["properties"].items()},
                   items=rules.get("items") or {}),
        channels=CHANNELS, units=CHANNEL_UNITS, game_version=g.version.get("VersionDate", "?"))


# ---------------------------------------------------------------------- engine (mirrored in itemvalue.js)
DMG_CH = ["spirit", "weapon", "melee"]             # ability damage, by what it scales with
KIT_CH = DMG_CH + ["heal", "hard_cc", "slow"]      # what one ability cast produces


def cast_uses(cd, charges, A: dict) -> float:
    """Casts per fight of an ability: one fight of combat_s per minute; it is recast cast_delay s
    after coming off cooldown; charges are spent together."""
    if not cd or cd <= 0:
        return 0.0
    ce, ch = cd + A["cast_delay"], max(1.0, charges or 1.0)
    return min(60 * ch / ce, max(ch, A["combat_s"] * ch / ce))


def renewal(cd, lam: float, T: float) -> float:
    """Expected procs in a fight of T s of an effect with cooldown cd that needs a trigger arriving
    at lam per second: each proc waits on average 1/lam s for its trigger after the cooldown."""
    if lam <= 0:
        return 0.0
    if not cd or cd <= 0:
        return lam * T
    ce = cd + 1 / lam
    return min((1 - np.exp(-lam * T)) * max(1.0, T / ce), 60 / ce, lam * T)


def charge_uses(cd, lam: float, T: float) -> float:
    """Full-charge equivalents per fight of a charge-up item (Mystic Burst, Tankbuster, ...): it starts the
    fight charged, then every trigger releases whatever has charged since, so damage accrues with time
    once triggers are frequent."""
    if lam <= 0 or not cd:
        return 0.0
    return (1 - np.exp(-lam * T)) * (1 + max(0.0, T - 2 / lam) / cd)


def gun(hero: dict, sp: float, boons: int, A: dict, st: dict) -> dict:
    """The hero's gun and melee at a stage: sustained shots/s (magazine + reload; Haze's magazine
    grows with spirit power), sustained weapon DPS, bullet hits on heroes per second of fight, the
    weapon damage bonus (fraction) and melee damage per light / heavy hit."""
    wbonus = st["item_weapon_pct"] / 100
    shots = (hero["clip"] + hero["clip_sp"] * sp) / hero["ammo"]
    rps = hero["rps"]
    rps_s = shots / (shots / rps + hero["reload"]) if hero["reload"] and shots > 0 else rps
    dps = (hero["dps"] + hero["ls_dps"] * boons) * (1 + wbonus) * rps_s / rps
    return dict(rps=rps_s, dps=dps, hits_ps=A["shoot_share"] * rps_s * A["hit_rate"], wbonus=wbonus,
                wbonus_pct=wbonus * 100, light=hero["light"] + hero["ls_light"] * boons,
                heavy=hero["heavy"] + hero["ls_heavy"] * boons,
                hp=hero["max_health"] + hero["ls_hp"] * boons + st["item_hp"],
                bullet=(hero["bullet"] + hero["ls_bullet"] * boons) * hero["bps"] * (1 + wbonus))


def ability_out(ab: dict, sp: float, g: dict, A: dict) -> dict:
    """One ability's own output: per cast (pc) and fight-level for passives / procs (fl), its spirit
    coefficients, casts per fight and gun-buff damage (amps are added by kit_stats)."""
    T = A["combat_s"]
    hits_ps = g["hits_ps"]
    base_hit = A["shoot_share"] * A["hit_rate"]
    passive = ab["passive"]
    uses = ab["uses"] if ab["uses"] is not None else cast_uses(ab["cd"], ab["charges"], A)
    pc = {c: 0.0 for c in KIT_CH}            # one cast
    fl = {c: 0.0 for c in KIT_CH}            # fight-level (passive hits, procs and buffs)
    pc_coef = fl_coef = buff_cast = buff_fl = 0.0
    ab_tg = ab["targets"] if ab["targets"] is not None else A["targets"]
    for t in ab["terms"] + ab["heal"]:
        chan = t["chan"]
        tg = ab_tg if t["aoe"] else 1.0
        m = (t["mult"] if t["mult"] is not None else A["default_dot_s"]) * (g[t["of"]] if t["of"] else 1.0)
        amt = (t["base"] + t["coef"] * sp + t["mcoef"] * g["light"] + t["hcoef"] * g["heavy"]) \
            * (1 + t["wcoef"] * g["wbonus"]) * m * tg
        c = t["coef"] * m * tg if chan == "spirit" else 0.0
        if t["per"] == "proc":                                  # procs on bullet hits / headshots
            lam = hits_ps * (A["headshot_share"] if t["on"] == "headshot" else 1.0) / t["hits_needed"]
            n = renewal(t["cd"], lam, T)
            fl[chan] += amt * n
            fl_coef += c * n
        elif t["per"] == "hit" and passive:                      # every bullet hit, all fight
            n = hits_ps * T
            fl[chan] += amt * n
            fl_coef += c * n
        else:
            n = hits_ps * min(t["window"] or A["default_dot_s"], T) if t["per"] == "hit" else 1.0
            pc[chan] += amt * n
            pc_coef += c * n
    for t in ab["gun"]:                       # boosted shooting: the extra over normal shooting
        hit = t["hit"] if t["hit"] is not None else base_hit
        boost = (1 + t["fire_pct"] / 100) * (1 + t["dmg_pct"] / 100) * hit * (ab_tg if t["aoe"] else 1.0)
        extra = g["dps"] * min(t["window"] if t["window"] else T, T) * (boost - base_hit)
        if passive:
            buff_fl += extra
        else:
            buff_cast += extra
    pc["weapon"] += buff_cast
    fl["weapon"] += buff_fl
    cc, tg = ab["cc"], (ab_tg if ab["aoe"] else 1.0)
    pc["hard_cc"] = (cc.get("stun", 0.0) + cc.get("silence", 0.0) * A["silence_weight"]
                     + cc.get("disarm", 0.0) * A["disarm_weight"] + cc.get("root", 0.0) * A["root_weight"]
                     + cc.get("blind", 0.0) * A["blind_weight"]) * tg
    pc["slow"] = cc.get("slow", 0.0) * tg
    return dict(ab=ab, passive=passive, uses=uses, pc=pc, fl=fl, pc_coef=pc_coef, fl_coef=fl_coef,
                buff=buff_cast * uses + buff_fl)


def add_amps(x: dict, dmg_ps: float, share: dict, g: dict, A: dict):
    """Damage amps / team buffs / damage reduction of one ability, in damage and healing terms:
    damage: +pct % of (the hero's own damage per second, standing in for each ally's) x window x heroes;
    defense: pct % of the incoming damage per second x window x heroes, as healing."""
    T = A["combat_s"]
    dst = x["fl"] if x["passive"] else x["pc"]
    for a in x["ab"]["amp"]:
        w = min(a["window"] if a["window"] else T, T)
        if a["kind"] == "defense":
            dst["heal"] += a["pct"] / 100 * g["incoming_dps"] * w * a["targets"]
        else:
            extra = a["pct"] / 100 * dmg_ps * w * a["targets"]
            for c in DMG_CH:
                dst[c] += extra * share[c]


def score_of(pc: dict, A: dict, cc_eq: float) -> float:
    """Rough worth of one cast, in damage: damage + healing + CC as the damage it prevents."""
    return sum(pc[c] for c in DMG_CH) + pc["heal"] + (pc["hard_cc"] + pc["slow"] * A["slow_to_stun"] / 100) * cc_eq


def kit_stats(abilities: list, sp: float, g: dict, A: dict, cc_eq: float, ults: list | None = None) -> dict:
    """What a hero's abilities produce per fight at `sp` spirit power: spirit, weapon and melee damage
    (each split into active casts vs passives), healing and CC, the spirit damage gained per point
    of spirit power, casts per fight, and what one cast of the best non-ultimate active (Echo Shard)
    and of the ultimate (Refresher) produces. Passives (no cooldown) are never echoed or refreshed.
    g: the hero's gun etc. at this stage -- hits_ps, dps (sustained), wbonus (fraction), light / heavy
    (melee damage per hit), hp, bullet (damage per shot), wbonus_pct, incoming_dps.
    ults: other heroes' ultimates, for an ability that copies one (copy_ult)."""
    T = A["combat_s"]
    abs_ = [ability_out(ab, sp, g, A) for ab in abilities]     # pass 1: each ability on its own

    def base_damage(xs):
        d = {c: sum(x["pc"][c] * x["uses"] + x["fl"][c] for x in xs) for c in DMG_CH}
        d["weapon"] += g["dps"] * A["shoot_share"] * A["hit_rate"] * T          # the gun
        tot = sum(d.values()) or 1.0
        return tot / T, {c: d[c] / tot for c in DMG_CH}
    dmg_ps, share = base_damage(abs_)
    for x in abs_:                            # pass 2: amps / team buffs on the hero's own damage rate
        add_amps(x, dmg_ps, share, g, A)
    for x in abs_:                            # pass 3: an ult that copies an enemy's (Sinclair)
        if x["ab"]["copy_ult"] is not None and ults:
            cands = []
            for u in ults:
                y = ability_out(u, sp, g, A)     # the copied ult with this hero's own stats
                add_amps(y, dmg_ps, share, g, A)
                cands.append((score_of(y["pc"], A, cc_eq), y))
            cands.sort(key=lambda s: s[0])
            pick = cands[int(np.floor(x["ab"]["copy_ult"] * (len(cands) - 1)))][1]
            for c in KIT_CH:
                x["pc"][c] += pick["pc"][c]
            x["pc_coef"] += pick["pc_coef"]
            x["buff"] += pick["buff"] / pick["uses"] * x["uses"] if pick["uses"] else 0.0
    for x in abs_:                            # pass 4: abilities that refresh the others (Yamato's ult)
        if x["ab"]["refresh"]:
            for y in abs_:
                if y is not x and not y["passive"] and not y["ab"]["ult"]:
                    for c in KIT_CH:
                        x["pc"][c] += y["pc"][c]
                    x["pc_coef"] += y["pc_coef"]
                    x["buff"] += (y["buff"] / y["uses"] if y["uses"] else 0.0) * x["uses"]
    tot = {c: 0.0 for c in DMG_CH}            # per fight, all abilities
    act = {c: 0.0 for c in DMG_CH}            # per fight, cast (active) abilities only
    dsp = casts = ult_uses = wd_buff = 0.0
    rows, echo, best, ult, echo_i = [], {c: 0.0 for c in KIT_CH}, -1.0, {c: 0.0 for c in KIT_CH}, -1
    for x in abs_:
        ab, passive, uses, pc, fl = x["ab"], x["passive"], x["uses"], x["pc"], x["fl"]
        casts += uses
        wd_buff += x["buff"]
        fight = {c: pc[c] * uses + fl[c] for c in DMG_CH}
        for c in DMG_CH:
            tot[c] += fight[c]
            if not passive:
                act[c] += pc[c] * uses
        coef = x["pc_coef"] * uses + x["fl_coef"]
        dsp += coef
        rows.append(dict(key=ab["key"], name=ab["name"], slot=ab["slot"], ult=ab["ult"], passive=passive, cd=ab["cd"],
                         uses=uses, dmg_cast=pc["spirit"], coef_cast=x["pc_coef"], dmg_fight=fight["spirit"],
                         coef_fight=coef, wdmg_cast=pc["weapon"], wdmg_fight=fight["weapon"], mdmg_cast=pc["melee"],
                         mdmg_fight=fight["melee"], heal_fight=pc["heal"] * uses + fl["heal"], cc_cast=pc["hard_cc"],
                         slow_cast=pc["slow"], src=ab["src"], note=ab["note"]))
        if not passive and not ab["ult"]:
            score = score_of(pc, A, cc_eq)
            if score > best:
                best, echo, echo_i = score, dict(pc), len(rows) - 1
        if ab["ult"] and not passive:
            ult, ult_uses = dict(pc), uses
    for i, r in enumerate(rows):
        r["echo"] = i == echo_i
    return dict(sd=tot["spirit"], wd=tot["weapon"], md=tot["melee"], sd_active=act["spirit"], wd_active=act["weapon"],
                wd_buff=wd_buff, dsd_dsp=dsp, casts=casts, echo=echo, ult=ult, ult_uses=ult_uses, rows=rows)


class Ref:
    """A hero (or the average of all heroes) at one game stage."""

    def __init__(self, A: dict, name: str, st: dict, hero: dict, boon_souls: list, avg_melee: float,
                 heroes: list | None = None):
        self.name, self.st, self.A, self.hero = name, st, A, hero["name"]
        self.souls = st["souls"]
        self.boons = sum(1 for x in boon_souls if x <= self.souls)
        self.base_hp = hero["max_health"] + hero["ls_hp"] * self.boons
        self.hp = self.base_hp + st["item_hp"]
        self.sp = hero["ls_sp"] * self.boons + st["item_spirit"]
        self.resist = {"bullet": st["bullet_resist"], "spirit": st["spirit_resist"], "melee": 0.0}
        self.mix = A["incoming_mix"]
        self.combat = T = A["combat_s"]
        wmult = 1 + st["item_weapon_pct"] / 100
        g = gun(hero, self.sp, self.boons, A, st)
        self.rps, self.dps, self.hits_ps = g["rps"], g["dps"], g["hits_ps"]
        self.bullet = (hero["bullet"] + hero["ls_bullet"] * self.boons) * hero["bps"] * wmult
        self.wd_gun = self.dps * T * A["shoot_share"] * A["hit_rate"]   # gun damage landed per fight
        self.md_base = st["melee_dmg_per_fight"] * hero["melee"] / avg_melee
        self.heal_base = st["heal_per_fight"]
        self.enemy_heal = st["enemy_heal_per_fight"]
        cc_eq = A["stun_dps_multiple"] * self.incoming_dps          # damage-equivalent of 1 s of stun
        g["incoming_dps"] = self.incoming_dps
        ults_of = lambda name: [a for h in heroes for a in h["abilities"]
                                if a["ult"] and not a["passive"] and a["copy_ult"] is None and h["name"] != name]
        if hero.get("abilities"):
            k = kit_stats(hero["abilities"], self.sp, g, A, cc_eq, ults_of(hero["name"]))
        else:                       # average hero: the mean over all heroes, each with its own gun and SP
            ks = []
            for h in heroes:
                sp = h["ls_sp"] * self.boons + st["item_spirit"]
                gh = gun(h, sp, self.boons, A, st)
                gh["incoming_dps"] = self.incoming_dps
                ks.append(kit_stats(h["abilities"], sp, gh, A, cc_eq, ults_of(h["name"])))
            n = len(ks)
            k = {f: sum(x[f] for x in ks) / n
                 for f in ("sd", "wd", "md", "sd_active", "wd_active", "wd_buff", "dsd_dsp", "casts", "ult_uses")}
            k.update(echo={c: sum(x["echo"][c] for x in ks) / n for c in KIT_CH},
                     ult={c: sum(x["ult"][c] for x in ks) / n for c in KIT_CH}, rows=None)
        self.sd, self.dsd_dsp, self.casts = k["sd"], k["dsd_dsp"], k["casts"]   # spirit dmg per fight, per SP
        self.sd_active, self.wd_active = k["sd_active"], k["wd_active"]          # from cast abilities only
        self.wd_kit, self.md_kit = k["wd"], k["md"]
        self.wd_shots = self.wd_gun + k["wd_buff"]   # bullets fired per fight (gun + ability gun buffs)
        self.wd = self.wd_gun + k["wd"]       # weapon damage per fight: bullets + weapon-scaled ability damage
        self.md = self.md_base + k["md"]      # melee damage per fight: melee hits + melee-scaled abilities
        self.echo, self.ult, self.ult_uses, self.kit_rows = k["echo"], k["ult"], k["ult_uses"], k["rows"]
        tpf = A["triggers_per_fight"]
        self.rate = {"hit": self.hits_ps, "headshot": self.hits_ps * A["headshot_share"], "cast": self.casts / T,
                     "imbued_cast": tpf["imbued_cast"] / T, "melee": tpf["melee"] / T}   # triggers per second

    def ehp(self, resist=None, hp=None):
        r = resist or self.resist
        return (hp if hp is not None else self.hp) / sum(self.mix[t] * (1 - r[t]) for t in self.mix)

    @property
    def incoming_dps(self):   # a fight costs `incoming_share` of your effective HP
        return self.ehp() * self.A["incoming_share"] / self.combat

    def describe(self):
        T = self.combat
        return dict(stage=self.name, hero=self.hero, souls=self.souls, boons=self.boons, hp=round(self.hp),
                    spirit_power=round(self.sp, 1), ehp=round(self.ehp()), weapon_dmg_per_fight=round(self.wd),
                    spirit_dmg_per_fight=round(self.sd), spirit_dmg_per_sp=round(self.dsd_dsp, 2),
                    bullet_dmg_per_shot=round(self.bullet, 1), incoming_dps=round(self.incoming_dps, 1),
                    gun_dmg_per_fight=round(self.wd_gun), ability_weapon_dmg_per_fight=round(self.wd_kit),
                    ability_melee_dmg_per_fight=round(self.md_kit), active_spirit_dmg_per_fight=round(self.sd_active),
                    combat_s=T, shots_per_s=round(self.rps, 2), hits_per_fight=round(self.hits_ps * T, 1),
                    headshots_per_fight=round(self.rate["headshot"] * T, 2), casts_per_fight=round(self.casts, 2))


class Engine:
    def __init__(self, inputs: dict, overrides: dict | None = None):
        self.I = inputs
        self.A = {**inputs["A"], **(overrides or {})}
        self.props = inputs["rules"]["props"]
        self.overrides = inputs["rules"]["items"]
        self.items = inputs["items"]
        self.by_key = {it["key"]: it for it in self.items}
        self.heroes = {h["id"]: h for h in inputs["heroes"]}
        self.avg = inputs["avg_hero"]

    def ref(self, stage: str, hero: dict | None = None) -> Ref:
        return Ref(self.A, stage, self.A["stages"][stage], hero or self.avg, self.I["boon_souls"], self.avg["melee"],
                   self.I["heroes"])

    # ------------------------------------------------------------------ helpers
    def _uses(self, cd, ref: Ref, trigger=None, per_shot_chance=None):
        """Activations per fight."""
        T = ref.combat
        if per_shot_chance is not None:   # on-hit procs: the hero's bullet hits in the fight x chance
            return min(ref.hits_ps * T * per_shot_chance, T / max(cd or 0.2, 0.2))
        if trigger:                       # needs an action (headshot, cast, melee hit) after each cooldown
            lam = ref.rate[trigger]
            return min(1.0, lam * T) if not cd or cd <= 0 else renewal(cd, lam, T)
        return 1.0 if not cd or cd <= 0 else min(60 / cd, max(1.0, T / cd))

    @staticmethod
    def _value(p: dict, ref: Ref) -> float:
        v = p["v"]
        if v is None:
            return 0.0
        if p["st"] == "spirit":
            v += p["sv"] * ref.sp
        elif p["st"] == "power_increase":
            v += p["sv"] * ref.boons
        return v

    def _apply(self, ch, kind, q, *, ref: Ref, uses, up, dur, aoe=False, held=None):
        A = self.A
        tg = A["targets"] if aoe else 1.0
        ooc_s = (60 - ref.combat) * A["ooc_efficiency"]      # useful out-of-combat seconds per minute
        if kind == "hp":
            ch["ehp"] += (ref.ehp(hp=ref.hp + q) - ref.ehp()) * up
        elif kind == "hp_pct":                                  # % of base (hero) health
            ch["ehp"] += (ref.ehp(hp=ref.hp + ref.base_hp * q / 100) - ref.ehp()) * up
        elif kind == "hp_pct_full":
            ch["ehp"] += ref.ehp() * q / 100 * min(1.0, uses)
        elif kind in ("resist_bullet", "resist_spirit", "resist_melee"):
            t = kind.split("_")[1]
            r2 = dict(ref.resist)
            r2[t] = 1 - (1 - r2[t]) * (1 - q / 100)
            ch["ehp"] += (ref.ehp(r2) - ref.ehp()) * up
        elif kind == "barrier":
            ch["ehp"] += q * uses
        elif kind == "immune_bullet_s":
            ch["ehp"] += ref.incoming_dps * ref.mix["bullet"] * q * min(1.0, uses)
        elif kind == "immune_all_s":
            ch["ehp"] += ref.incoming_dps * q * min(1.0, uses)
        elif kind == "regen":                                   # HP/s, always on
            ch["heal"] += q * ref.combat * up
            ch["recovery"] += q * ooc_s * up
        elif kind == "regen_proc":                              # HP/s while triggered in a fight
            ch["heal"] += q * ref.combat * up * tg
        elif kind == "ooc_regen":
            ch["recovery"] += q * ooc_s
        elif kind == "ooc_regen_pct":
            ch["recovery"] += q / 100 * ref.hp * ooc_s * 0.5
        elif kind == "heal":
            ch["heal"] += q * uses * tg
        elif kind == "recovery_heal":                           # heal that only works out of combat
            ch["recovery"] += q * min(uses, 1.0)
        elif kind == "heal_pct_hp":
            ch["heal"] += q / 100 * ref.hp * uses
        elif kind == "lifesteal_bullet":              # bullets only
            ch["heal"] += q / 100 * ref.wd_shots * up
        elif kind == "lifesteal_spirit":
            ch["heal"] += q / 100 * ref.sd * up
        elif kind == "heal_amp":
            ch["heal"] += q / 100 * ref.heal_base * up
        elif kind == "weapon_pct":                    # weapon damage %: bullets and weapon-scaled abilities
            ch["weapon"] += q / 100 * ref.wd * up
        elif kind == "fire_rate_pct":                 # fire rate / pellets / reload: bullets only
            ch["weapon"] += q / 100 * ref.wd_shots * up
        elif kind == "self_damage_pct":
            ch["weapon"] += q / 100 * ref.wd * up
            ch["spirit"] += q / 100 * ref.sd * up
        elif kind == "clip_pct":
            ch["clip"] += q * up
        elif kind == "bullet_speed_pct":
            ch["bullet_speed"] += q * up
        elif kind == "bullet_proc":
            ch["weapon"] += q * uses
        elif kind == "bullet_proc_pct":
            ch["weapon"] += q / 100 * ref.bullet * uses
        elif kind == "melee_pct":
            ch["melee"] += q / 100 * ref.md * up
        elif kind == "spirit_power":
            ch["spirit"] += q * ref.dsd_dsp * up
        elif kind == "spirit_power_pct":
            ch["spirit"] += q / 100 * ref.sp * ref.dsd_dsp * up
        elif kind == "spirit_dmg":
            ch["spirit"] += q * uses * tg
        elif kind == "spirit_dps":
            ch["spirit"] += q * (min(dur, ref.combat) * uses if dur else ref.combat * up) * tg
        elif kind == "spirit_pct_hp_dps":
            ch["spirit"] += q / 100 * ref.hp * (min(dur, ref.combat) * uses if dur else ref.combat * up) * tg
        elif kind == "spirit_pct_hp":
            ch["spirit"] += q / 100 * ref.hp * uses * tg
        elif kind == "spirit_amp_proc":               # amplifies one cast: passives are not cast
            ch["spirit"] += q / 100 * (ref.sd_active / ref.casts if ref.casts else 0.0) * uses
        elif kind == "cdr":                           # more casts of active abilities (passives have no cooldown)
            f = (1 / (1 - min(q, 90) / 100) - 1) * A["cdr_damage_share"] * up
            ch["spirit"] += ref.sd_active * f
            ch["weapon"] += ref.wd_active * f
        elif kind == "duration_pct":                  # stacks diminishingly: 1 - (1-a)(1-b)...
            f = q / 100 * (1 - ref.st.get("item_duration_pct", 0) / 100) * A["duration_damage_share"] * up
            ch["spirit"] += ref.sd_active * f
            ch["weapon"] += ref.wd_active * f
        elif kind == "enemy_bullet_resist_down":      # card values are negative (-10 = -10% enemy resist)
            ch["weapon"] += -q / 100 * ref.wd * (1 + A["team_amp"]) * up
        elif kind == "enemy_spirit_resist_down":
            ch["spirit"] += -q / 100 * ref.sd * (1 + A["team_amp"]) * up
        elif kind == "enemy_damage_down":
            ch["ehp"] += -q / 100 * ref.incoming_dps * ref.combat * up
        elif kind == "enemy_spirit_damage_down":
            ch["ehp"] += -q / 100 * ref.incoming_dps * ref.mix["spirit"] * ref.combat * up
        elif kind == "enemy_fire_rate_down":          # card values are positive (28 = -28% enemy fire rate)
            ch["ehp"] += q / 100 * ref.incoming_dps * ref.mix["bullet"] * ref.combat * up
        elif kind in ("reflect_bullet", "reflect_spirit"):   # share of incoming damage reflected while active
            t = "bullet" if kind == "reflect_bullet" else "spirit"
            ch["spirit"] += q / 100 * ref.incoming_dps * ref.mix[t] * \
                (min(dur, ref.combat) * uses if dur else ref.combat * up)
        elif kind == "self_drain":                    # HP/s lost while toggled on
            ch["heal"] -= q * ref.combat * up
        elif kind == "enemy_heal_down":
            ch["heal_denied"] += -q / 100 * ref.enemy_heal * up
        elif kind == "slow":                          # slow-%-seconds
            ch["slow"] += abs(q) * (min(dur, ref.combat) * uses if dur else ref.combat * up)
        elif kind in ("stun_s", "silence_s", "disarm_s"):
            wgt = {"stun_s": 1.0, "silence_s": A["silence_weight"], "disarm_s": A["disarm_weight"]}[kind]
            ch["hard_cc"] += q * uses * wgt * tg
        elif kind == "move":
            ch["move"] += q * up
        elif kind == "sprint":
            ch["sprint"] += q * up
        elif kind == "stamina":
            ch["stamina"] += q * up
        elif kind == "range_pct":                     # stacks diminishingly like duration
            ch["range"] += q * (1 - ref.st.get("item_range_pct", 0) / 100) * up
        elif kind == "item_cdr":                      # recharges the build's cooldown items (not charge-up ones)
            ch["items"] += ref.st.get("cd_item_value", 0) * (1 / (1 - min(q, 90) / 100) - 1) * up
        elif kind == "charges":
            ch["charges"] += q
        elif kind == "status_resist":
            ch["status_resist"] += q * up
        elif kind == "souls_per_min":
            ch["souls"] += q * A["remaining_minutes"]
        elif kind == "souls_held":                    # souls per minute while alive, for the minutes it is held
            ch["souls"] += q * (held if held else A["remaining_minutes"]) * A["alive_share"]
        elif kind == "echo":                          # a second cast of the best non-ultimate ability
            for c, v in ref.echo.items():
                ch[c] += v * uses * q
        elif kind == "refresh_ult":                   # a second ultimate, weighted for teamfight impact
            n = min(uses, ref.ult_uses)
            for c, v in ref.ult.items():
                ch[c] += v * n * A["ult_weight"] * q
        else:
            raise KeyError(f"unknown kind {kind}")

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, key: str, ref: Ref):
        """-> (channel quantities, unmodeled property keys, note)"""
        A = self.A
        it = self.by_key.get(key) or {"blocks": [], "item_cd": None}
        ov = self.overrides.get(key) or {}
        skip, remap = set(ov.get("skip") or []), ov.get("remap") or {}
        trigger = ov.get("trigger")
        held = ov.get("held_minutes") if isinstance(ov.get("held_minutes"), (int, float)) else None
        ch = defaultdict(float)
        unmodeled = []
        for b in it["blocks"]:
            btype, cd, dur, pval = b["t"], b["cd"], b["dur"], b["pval"]
            chance = b["pc"] / 100 if (ov.get("uses_per_min") == "per_shot" and b["pc"]) else None
            uses = self._uses(cd, ref, trigger, chance)
            if ov.get("charge_up") and trigger and cd:   # partial procs: damage scales with the charge
                uses = charge_uses(cd, ref.rate[trigger], ref.combat)
            if isinstance(ov.get("uses_per_min"), (int, float)):
                uses = float(ov["uses_per_min"])
            bu = pval.get("BuildUpPerShot")        # build-up procs: the same time to proc for every hero
            up_build = max(0.0, 1 - A["buildup_ref_s"] / bu / ref.combat) if bu else None
            for p in b["props"]:
                k = p["k"]
                if k in skip:
                    continue
                rule = dict(self.props.get(k) or {})
                if k in remap:
                    rule["kind"] = remap[k]
                if not rule:
                    unmodeled.append(k)
                    continue
                kind = rule["kind"]
                if kind == "ignore":
                    continue
                q = self._value(p, ref) * rule.get("factor", 1.0)
                if q == 0:
                    continue
                d = pval.get(rule["dur_key"]) if rule.get("dur_key") else dur
                cond = p["c"]
                if btype == "Innate" and not cd:
                    up = A["cond_uptime"] if cond else 1.0
                elif cd and kind not in INSTANT and kind not in OVER_DURATION:
                    up = min(1.0, uses * (d or min(cd, ref.combat) * 0.5) / ref.combat)
                elif not cd and btype in ("Passive", None, "Active"):
                    up = A["cond_uptime"] if (cond or d or btype == "Active") else 1.0
                else:
                    up = 1.0
                if up_build is not None and kind not in INSTANT:
                    up = up_build
                if isinstance(ov.get("uptime"), (int, float)):
                    up = float(ov["uptime"])
                c = rule.get("cond")
                cm = 1.0                    # condition share, applied to uptime and to activations
                if c == "low_hp":
                    cm = A["low_hp_uptime"] / (A["cond_uptime"] if up < 1 else 1)
                elif c == "close":
                    cm = A["close_share"]
                elif c == "long":
                    cm = A["long_share"]
                if rule.get("per_stack"):
                    q *= b["stacks"] * A["stack_fill"]
                self._apply(ch, kind, q, ref=ref, uses=uses * cm, up=up * cm, dur=d, aoe=rule.get("aoe", False),
                            held=held)
        for e in ov.get("add") or []:
            if e["kind"] == "sell_refund":            # sold after held_minutes: part of the cost comes back
                ch["souls"] += A["sell_refund"] * it.get("cost", 0.0)
                continue
            cd = _num(e.get("cooldown")) or it["item_cd"]
            use = e.get("use", "always")
            uses = self._uses(cd, ref, e.get("trigger", trigger)) if use != "always" else 1.0
            d = e.get("duration")
            up = 1.0 if use == "always" or not d else min(1.0, uses * d / ref.combat)
            self._apply(ch, e["kind"], float(e["value"]) * e.get("factor", 1.0), ref=ref, uses=uses, up=up,
                        dur=d, aoe=e.get("aoe", False), held=held)
        if held:                                      # stats held for part of the usual horizon count pro rata
            f = held / A["item_horizon_minutes"]
            for c in ch:
                if c != "souls":
                    ch[c] *= f
        return {k: v for k, v in ch.items() if v}, sorted(set(unmodeled)), ov.get("note")

    # ------------------------------------------------------------------ pricing
    def prices(self, ref: Ref) -> dict:
        A = self.A
        price = {"souls": 1.0}
        for chan, key in A["calibrate"]:
            if chan == "ehp":
                price["heal"] = None        # placeholder; heal is priced like ehp
            it = self.by_key.get(key)
            if it is None:
                log.warning("calibration item %s missing; %s uses the fallback price", key, chan)
                continue
            q, _, _ = self.evaluate(key, ref)
            known = sum(v * price[c] for c, v in q.items() if price.get(c) and c != chan)
            amount = q.get(chan, 0)
            left = it["cost"] - known
            if amount > 0 and left > 0:
                price[chan] = left / amount
                if chan == "ehp":
                    price["heal"] = price["ehp"]
            else:
                log.debug("stage %s: cannot calibrate %s from %s (amount %.3g, souls left %.0f)",
                          ref.name, chan, key, amount, left)
        if price.get("ehp"):
            # disabling an enemy for 1 s ~ the damage they would have dealt, x stun_dps_multiple
            price.setdefault("hard_cc", A["stun_dps_multiple"] * ref.incoming_dps * price["ehp"])
            price.setdefault("slow", price["hard_cc"] * A["slow_to_stun"] / 100)
        if price.get("weapon") and price.get("spirit"):
            price.setdefault("heal_denied", (price["weapon"] + price["spirit"]) / 2)
        for c, v in (A.get("fallback_prices") or {}).items():
            if not price.get(c):
                price[c] = float(v)
        return {c: v for c, v in price.items() if v}

    def run(self, hero_id: int | None = None, pricing: str = "average", stages=None):
        """Rows for every item at every stage. hero_id None = the average hero. pricing "average"
        prices the hero's quantities at the average hero's rates; "own" calibrates on the hero."""
        hero = self.heroes[hero_id] if hero_id else None
        rows, long = [], []
        ref_keys = {k for _, k in self.A["calibrate"]}
        meta = {"refs": {}, "prices": {}, "assumptions": {k: v for k, v in self.A.items() if k != "stages"},
                "stages": self.A["stages"], "units": CHANNEL_UNITS}
        tier_stage = {int(k): v for k, v in self.A["stage_by_tier"].items()}
        for sname in self.A["stages"]:
            if stages and sname not in stages:
                continue
            ref = self.ref(sname, hero)
            price = self.prices(ref if (hero is None or pricing == "own") else self.ref(sname))
            meta["refs"][sname] = ref.describe()
            meta["prices"][sname] = price
            if ref.kit_rows:
                meta.setdefault("kit", {})[sname] = ref.kit_rows
            for it in self.items:
                q, unm, note = self.evaluate(it["key"], ref)
                vals = {c: q.get(c, 0.0) * price.get(c, 0.0) for c in CHANNELS}
                value = sum(vals[c] for c in CHANNELS)
                rows.append(dict(
                    item_id=it["item_id"], key=it["key"], name=it["name"], slot=it["slot"], tier=it["tier"],
                    cost=it["cost"], activation=it["activation"], stage=sname,
                    tier_stage=tier_stage.get(it["tier"]) == sname,
                    value=value, roi=value / it["cost"] if it["cost"] else np.nan,
                    **{f"q_{c}": q.get(c, 0.0) for c in CHANNELS}, **{f"v_{c}": vals[c] for c in CHANNELS},
                    unmodeled=", ".join(unm), n_unmodeled=len(unm), note=note, reference=it["key"] in ref_keys))
                for c in CHANNELS:
                    if q.get(c):
                        long.append(dict(item_id=it["item_id"], name=it["name"], stage=sname, channel=c, quantity=q[c],
                                         per_1k_souls=q[c] / it["cost"] * 1000, souls=vals[c]))
        df = pd.DataFrame(rows)
        vc = df[[f"v_{c}" for c in CHANNELS]]
        df["main_channel"] = vc.idxmax(axis=1).str[2:].where(vc.max(axis=1) > 0)
        return df, pd.DataFrame(long), meta


def sensitivity(inputs: dict, base: pd.DataFrame) -> pd.DataFrame:
    """ROI under alternative assumptions (settings item_value.scenarios), tier stages only."""
    out = []
    for name, ov in (inputs["A"].get("scenarios") or {}).items():
        df, _, _ = Engine(inputs, ov).run()
        df = df[df.tier_stage][["item_id", "roi"]].rename(columns={"roi": f"roi_{name}"})
        out.append(df.set_index("item_id"))
    return pd.concat(out, axis=1).reset_index() if out else pd.DataFrame({"item_id": base.item_id.unique()})


def run(s: Settings, check: bool = False):
    inputs = build_inputs(s)
    df, long, meta = Engine(inputs).run()
    sens = sensitivity(inputs, df)
    df = df.merge(sens, on="item_id", how="left")
    meta["scenarios"] = inputs["A"].get("scenarios") or {}
    out = s.data_dir / "results"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "item_value.parquet", index=False)
    long.to_parquet(out / "item_value_channels.parquet", index=False)
    (out / "item_value.meta.json").write_text(json.dumps(meta, indent=1, default=float))
    (out / "item_value_inputs.json").write_text(json.dumps(inputs, separators=(",", ":"), default=float))
    log.info("wrote item_value (%d rows, %d heroes); average hero: %s", len(df), len(inputs["heroes"]), meta["refs"])
    if check:
        check_js(inputs)
    return df, long, meta


# ---------------------------------------------------------------------- Python <-> JS parity
def check_js(inputs: dict, tol: float = 1e-6) -> bool:
    """Run itemvalue.js under node on the same inputs and compare every q_/v_/roi value."""
    node = shutil.which("node")
    if not node:
        log.warning("node not found; skipping the itemvalue.js check")
        return False
    names = {h["name"]: h["id"] for h in inputs["heroes"]}
    cases = [dict(hero=None, pricing="average", overrides={}),
             dict(hero=names.get("Mina"), pricing="average", overrides={}),
             dict(hero=names.get("Venator"), pricing="own", overrides={}),
             dict(hero=names.get("Haze"), pricing="average",
                  overrides={"combat_s": 25, "targets": 3.0, "headshot_share": 0.35, "cast_delay": 0.5,
                             "triggers_per_fight": {**inputs["A"]["triggers_per_fight"], "melee": 4}}),
             dict(hero=names.get("Wraith"), pricing="average", overrides={"combat_s": 8, "ult_weight": 3.0}),
             dict(hero=names.get("Sinclair"), pricing="own", overrides={"targets": 2.0}),
             dict(hero=names.get("Rat King"), pricing="average", overrides={})]
    cols = [f"{p}_{c}" for c in CHANNELS for p in ("q", "v")] + ["value", "roi"]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "inputs.json").write_text(json.dumps(inputs, default=float))
        (tdp / "cases.json").write_text(json.dumps(cases))
        script = ("const fs=require('fs');const IV=require(process.argv[1]);"
                  "const I=JSON.parse(fs.readFileSync(process.argv[2]));const C=JSON.parse(fs.readFileSync(process.argv[3]));"
                  "process.stdout.write(JSON.stringify(C.map(c=>IV.run(I,{hero:c.hero,pricing:c.pricing,overrides:c.overrides}).rows)));")
        r = subprocess.run([node, "-e", script, str(JS), str(tdp / "inputs.json"), str(tdp / "cases.json")],
                           capture_output=True, text=True, timeout=120)
    if r.returncode:
        log.error("itemvalue.js failed:\n%s", r.stderr)
        return False
    ok = True
    for case, js_rows in zip(cases, json.loads(r.stdout)):
        py, _, _ = Engine(inputs, case["overrides"]).run(case["hero"], case["pricing"])
        js = pd.DataFrame(js_rows)
        m = py.merge(js, on=["item_id", "stage"], suffixes=("", "_js"))
        if len(m) != len(py) or len(js) != len(py):
            log.error("row count differs for %s: py %d js %d", case, len(py), len(js))
            ok = False
        for c in cols:
            a, b = m[c].astype(float).fillna(0), m[f"{c}_js"].astype(float).fillna(0)
            bad = (a - b).abs() > tol * np.maximum(1, a.abs())
            if bad.any():
                ok = False
                x = m[bad].iloc[0]
                log.error("%s: %s differs on %d rows (e.g. %s %s: py %.6g js %.6g)", case, c, bad.sum(), x["name"],
                          x["stage"], x[c], x[f"{c}_js"])
        if (m.main_channel.fillna("") != m.main_channel_js.fillna("")).any():
            ok = False
            log.error("%s: main_channel differs", case)
    log.info("itemvalue.js parity check: %s", "OK" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings")
    ap.add_argument("--check", action="store_true", help="compare the browser engine (itemvalue.js) with this one")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_settings(a.settings), check=a.check)


if __name__ == "__main__":
    main()
