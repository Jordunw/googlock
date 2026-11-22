// Item value engine for the browser: a line-for-line mirror of dla/itemvalue.py (Ref, Engine).
// Inputs come from item_value_inputs.json (python -m dla.itemvalue). Keep both in step;
// `python -m dla.itemvalue --check` runs this file under node and compares every number.
//   IV.run(inputs, { hero: heroId | null, pricing: "average" | "own", overrides: {...} })
//     -> { rows: [...one per item x stage...], meta: { refs, prices, kit: per-ability rows for a hero } }
(function (root) {
  "use strict";
  const CHANNELS = ["ehp", "heal", "recovery", "weapon", "spirit", "melee", "heal_denied", "slow", "hard_cc", "clip",
    "bullet_speed", "move", "sprint", "stamina", "range", "charges", "status_resist", "souls", "items"];
  const INSTANT = new Set(["heal", "heal_pct_hp", "recovery_heal", "barrier", "spirit_dmg", "spirit_pct_hp", "bullet_proc",
    "bullet_proc_pct", "stun_s", "silence_s", "disarm_s", "hp_pct_full", "immune_bullet_s", "immune_all_s", "spirit_amp_proc"]);
  const OVER_DURATION = new Set(["spirit_dps", "spirit_pct_hp_dps", "slow", "reflect_bullet", "reflect_spirit"]);
  const isNum = v => typeof v === "number" && isFinite(v);
  const truthy = v => v !== null && v !== undefined && v !== 0 && v !== false && v !== "" && !Number.isNaN(v);
  const num = v => {                                     // _num: numbers, or the leading number of a string
    if (typeof v === "number") return v;
    if (typeof v === "string") { const m = /^\s*(-?[\d.]+)/.exec(v); const x = m ? parseFloat(m[1]) : NaN; return isNaN(x) ? null : x; }
    return null;
  };

  const DMG_CH = ["spirit", "weapon", "melee"];
  const KIT_CH = [...DMG_CH, "heal", "hard_cc", "slow"];
  const zeros = () => Object.fromEntries(KIT_CH.map(c => [c, 0.0]));
  function castUses(cd, charges, A) {               // casts per fight of an ability
    if (!truthy(cd) || cd <= 0) return 0.0;
    const ce = cd + A.cast_delay, ch = Math.max(1.0, truthy(charges) ? charges : 1.0);
    return Math.min(60 * ch / ce, Math.max(ch, A.combat_s * ch / ce));
  }
  function renewal(cd, lam, T) {                    // procs per fight: cooldown + wait for the next trigger
    if (lam <= 0) return 0.0;
    if (!truthy(cd) || cd <= 0) return lam * T;
    const ce = cd + 1 / lam;
    return Math.min((1 - Math.exp(-lam * T)) * Math.max(1.0, T / ce), 60 / ce, lam * T);
  }
  function chargeUses(cd, lam, T) {                 // charge-up items: damage accrues with time charged
    if (lam <= 0 || !truthy(cd)) return 0.0;
    return (1 - Math.exp(-lam * T)) * (1 + Math.max(0.0, T - 2 / lam) / cd);
  }
  function gun(hero, sp, boons, A, st) {            // the hero's gun and melee at a stage
    const wbonus = st.item_weapon_pct / 100;
    const shots = (hero.clip + hero.clip_sp * sp) / hero.ammo, rps = hero.rps;
    const rpsS = truthy(hero.reload) && shots > 0 ? shots / (shots / rps + hero.reload) : rps;
    const dps = (hero.dps + hero.ls_dps * boons) * (1 + wbonus) * rpsS / rps;
    return { rps: rpsS, dps, hits_ps: A.shoot_share * rpsS * A.hit_rate, wbonus, wbonus_pct: wbonus * 100,
      light: hero.light + hero.ls_light * boons, heavy: hero.heavy + hero.ls_heavy * boons,
      hp: hero.max_health + hero.ls_hp * boons + st.item_hp,
      bullet: (hero.bullet + hero.ls_bullet * boons) * hero.bps * (1 + wbonus) };
  }
  function abilityOut(ab, sp, g, A) {              // one ability's own output (amps added later)
    const T = A.combat_s, hitsPs = g.hits_ps, baseHit = A.shoot_share * A.hit_rate;
    const passive = ab.passive;
    const uses = ab.uses !== null && ab.uses !== undefined ? ab.uses : castUses(ab.cd, ab.charges, A);
    const pc = zeros(), fl = zeros();
    let pcCoef = 0, flCoef = 0, buffCast = 0, buffFl = 0;
    const abTg = ab.targets !== null && ab.targets !== undefined ? ab.targets : A.targets;
    for (const t of [...ab.terms, ...ab.heal]) {
      const chan = t.chan, tg = t.aoe ? abTg : 1.0;
      const m = (t.mult !== null && t.mult !== undefined ? t.mult : A.default_dot_s) * (t.of ? g[t.of] : 1.0);
      const amt = (t.base + t.coef * sp + t.mcoef * g.light + t.hcoef * g.heavy) * (1 + t.wcoef * g.wbonus) * m * tg;
      const c = chan === "spirit" ? t.coef * m * tg : 0.0;
      if (t.per === "proc") {
        const lam = hitsPs * (t.on === "headshot" ? A.headshot_share : 1.0) / t.hits_needed;
        const n = renewal(t.cd, lam, T);
        fl[chan] += amt * n; flCoef += c * n;
      } else if (t.per === "hit" && passive) {
        const n = hitsPs * T;
        fl[chan] += amt * n; flCoef += c * n;
      } else {
        const n = t.per === "hit" ? hitsPs * Math.min(truthy(t.window) ? t.window : A.default_dot_s, T) : 1.0;
        pc[chan] += amt * n; pcCoef += c * n;
      }
    }
    for (const t of ab.gun) {
      const hit = t.hit !== null && t.hit !== undefined ? t.hit : baseHit;
      const boost = (1 + t.fire_pct / 100) * (1 + t.dmg_pct / 100) * hit * (t.aoe ? abTg : 1.0);
      const extra = g.dps * Math.min(truthy(t.window) ? t.window : T, T) * (boost - baseHit);
      if (passive) buffFl += extra; else buffCast += extra;
    }
    pc.weapon += buffCast; fl.weapon += buffFl;
    const cc = ab.cc, tg = ab.aoe ? abTg : 1.0;
    pc.hard_cc = ((cc.stun || 0) + (cc.silence || 0) * A.silence_weight + (cc.disarm || 0) * A.disarm_weight
      + (cc.root || 0) * A.root_weight + (cc.blind || 0) * A.blind_weight) * tg;
    pc.slow = (cc.slow || 0) * tg;
    return { ab, passive, uses, pc, fl, pcCoef, flCoef, buff: buffCast * uses + buffFl };
  }
  function addAmps(x, dmgPs, share, g, A) {          // damage amps / team buffs / damage reduction
    const T = A.combat_s, dst = x.passive ? x.fl : x.pc;
    for (const a of x.ab.amp || []) {
      const w = Math.min(truthy(a.window) ? a.window : T, T);
      if (a.kind === "defense") dst.heal += a.pct / 100 * g.incoming_dps * w * a.targets;
      else {
        const extra = a.pct / 100 * dmgPs * w * a.targets;
        for (const c of DMG_CH) dst[c] += extra * share[c];
      }
    }
  }
  const scoreOf = (pc, A, ccEq) => { let s = 0; for (const c of DMG_CH) s += pc[c];
    return s + pc.heal + (pc.hard_cc + pc.slow * A.slow_to_stun / 100) * ccEq; };
  function kitStats(abilities, sp, g, A, ccEq, ults) {
    const T = A.combat_s;
    const abs = abilities.map(ab => abilityOut(ab, sp, g, A));     // pass 1
    const d = {};
    for (const c of DMG_CH) { d[c] = 0; for (const x of abs) d[c] += x.pc[c] * x.uses + x.fl[c]; }
    d.weapon += g.dps * A.shoot_share * A.hit_rate * T;
    let totD = 0; for (const c of DMG_CH) totD += d[c];
    if (!totD) totD = 1.0;
    const dmgPs = totD / T, share = {};
    for (const c of DMG_CH) share[c] = d[c] / totD;
    for (const x of abs) addAmps(x, dmgPs, share, g, A);           // pass 2
    for (const x of abs) {                                         // pass 3: copied ult (Sinclair)
      if (x.ab.copy_ult === null || x.ab.copy_ult === undefined || !ults || !ults.length) continue;
      const cands = ults.map(u => { const y = abilityOut(u, sp, g, A); addAmps(y, dmgPs, share, g, A);
        return [scoreOf(y.pc, A, ccEq), y]; });
      cands.sort((p, q) => p[0] - q[0]);
      const pick = cands[Math.floor(x.ab.copy_ult * (cands.length - 1))][1];
      for (const c of KIT_CH) x.pc[c] += pick.pc[c];
      x.pcCoef += pick.pcCoef;
      x.buff += pick.uses ? pick.buff / pick.uses * x.uses : 0.0;
    }
    for (const x of abs) {                                         // pass 4: refreshes (Yamato)
      if (!x.ab.refresh) continue;
      for (const y of abs) {
        if (y === x || y.passive || y.ab.ult) continue;
        for (const c of KIT_CH) x.pc[c] += y.pc[c];
        x.pcCoef += y.pcCoef;
        x.buff += (y.uses ? y.buff / y.uses : 0.0) * x.uses;
      }
    }
    const tot = { spirit: 0, weapon: 0, melee: 0 }, act = { spirit: 0, weapon: 0, melee: 0 };
    let dsp = 0, casts = 0, ultUses = 0, best = -1.0, wdBuff = 0.0, echoI = -1;
    const rows = [];
    let echo = zeros(), ult = zeros();
    for (const x of abs) {
      const { ab, passive, uses, pc, fl } = x;
      casts += uses; wdBuff += x.buff;
      const fight = {};
      for (const c of DMG_CH) {
        fight[c] = pc[c] * uses + fl[c];
        tot[c] += fight[c];
        if (!passive) act[c] += pc[c] * uses;
      }
      const coef = x.pcCoef * uses + x.flCoef;
      dsp += coef;
      rows.push({ key: ab.key, name: ab.name, slot: ab.slot, ult: ab.ult, passive, cd: ab.cd, uses, dmg_cast: pc.spirit,
        coef_cast: x.pcCoef, dmg_fight: fight.spirit, coef_fight: coef, wdmg_cast: pc.weapon, wdmg_fight: fight.weapon,
        mdmg_cast: pc.melee, mdmg_fight: fight.melee, heal_fight: pc.heal * uses + fl.heal, cc_cast: pc.hard_cc,
        slow_cast: pc.slow, src: ab.src, note: ab.note });
      if (!passive && !ab.ult) {
        const score = scoreOf(pc, A, ccEq);
        if (score > best) { best = score; echo = Object.assign({}, pc); echoI = rows.length - 1; }
      }
      if (ab.ult && !passive) { ult = Object.assign({}, pc); ultUses = uses; }
    }
    rows.forEach((r, i) => { r.echo = i === echoI; });
    return { sd: tot.spirit, wd: tot.weapon, md: tot.melee, sd_active: act.spirit, wd_active: act.weapon,
      wd_buff: wdBuff, dsd_dsp: dsp, casts, echo, ult, ult_uses: ultUses, rows };
  }

  class Ref {
    constructor(A, name, st, hero, boonSouls, avgMelee, heroes) {
      this.name = name; this.st = st; this.A = A; this.hero = hero.name;
      this.souls = st.souls;
      this.boons = boonSouls.filter(x => x <= this.souls).length;
      this.base_hp = hero.max_health + hero.ls_hp * this.boons;
      this.hp = this.base_hp + st.item_hp;
      this.sp = hero.ls_sp * this.boons + st.item_spirit;
      this.resist = { bullet: st.bullet_resist, spirit: st.spirit_resist, melee: 0.0 };
      this.mix = A.incoming_mix;
      const T = this.combat = A.combat_s;
      const wmult = 1 + st.item_weapon_pct / 100;
      const g = gun(hero, this.sp, this.boons, A, st);
      this.rps = g.rps; this.dps = g.dps; this.hits_ps = g.hits_ps;
      this.bullet = (hero.bullet + hero.ls_bullet * this.boons) * hero.bps * wmult;
      this.wd_gun = this.dps * T * A.shoot_share * A.hit_rate;
      this.md_base = st.melee_dmg_per_fight * hero.melee / avgMelee;
      this.heal_base = st.heal_per_fight;
      this.enemy_heal = st.enemy_heal_per_fight;
      const ccEq = A.stun_dps_multiple * this.incoming_dps;
      g.incoming_dps = this.incoming_dps;
      const ultsOf = name => heroes.flatMap(h => h.name === name ? [] :
        h.abilities.filter(a => a.ult && !a.passive && (a.copy_ult === null || a.copy_ult === undefined)));
      let k;
      if (hero.abilities) k = kitStats(hero.abilities, this.sp, g, A, ccEq, ultsOf(hero.name));
      else {                                         // average hero: mean over all heroes
        const ks = heroes.map(h => {
          const sp = h.ls_sp * this.boons + st.item_spirit;
          const gh = gun(h, sp, this.boons, A, st);
          gh.incoming_dps = this.incoming_dps;
          return kitStats(h.abilities, sp, gh, A, ccEq, ultsOf(h.name));
        });
        const n = ks.length, mean = f => ks.reduce((a, x) => a + x[f], 0) / n;
        k = { rows: null };
        for (const f of ["sd", "wd", "md", "sd_active", "wd_active", "wd_buff", "dsd_dsp", "casts", "ult_uses"]) k[f] = mean(f);
        k.echo = Object.fromEntries(KIT_CH.map(c => [c, ks.reduce((a, x) => a + x.echo[c], 0) / n]));
        k.ult = Object.fromEntries(KIT_CH.map(c => [c, ks.reduce((a, x) => a + x.ult[c], 0) / n]));
      }
      this.sd = k.sd; this.dsd_dsp = k.dsd_dsp; this.casts = k.casts;
      this.sd_active = k.sd_active; this.wd_active = k.wd_active;
      this.wd_kit = k.wd; this.md_kit = k.md;
      this.wd_shots = this.wd_gun + k.wd_buff;
      this.wd = this.wd_gun + k.wd;
      this.md = this.md_base + k.md;
      this.echo = k.echo; this.ult = k.ult; this.ult_uses = k.ult_uses; this.kit_rows = k.rows;
      const tpf = A.triggers_per_fight;
      this.rate = { hit: this.hits_ps, headshot: this.hits_ps * A.headshot_share, cast: this.casts / T,
        imbued_cast: tpf.imbued_cast / T, melee: tpf.melee / T };
    }
    ehp(resist, hp) {
      const r = resist || this.resist;
      let d = 0; for (const t of Object.keys(this.mix)) d += this.mix[t] * (1 - r[t]);
      return (hp !== undefined && hp !== null ? hp : this.hp) / d;
    }
    get incoming_dps() { return this.ehp() * this.A.incoming_share / this.combat; }
    describe() {
      const r = (v, d = 0) => Math.round(v * 10 ** d) / 10 ** d, T = this.combat;
      return { stage: this.name, hero: this.hero, souls: this.souls, boons: this.boons, hp: r(this.hp), spirit_power: r(this.sp, 1),
        ehp: r(this.ehp()), weapon_dmg_per_fight: r(this.wd), spirit_dmg_per_fight: r(this.sd), spirit_dmg_per_sp: r(this.dsd_dsp, 2),
        bullet_dmg_per_shot: r(this.bullet, 1), incoming_dps: r(this.incoming_dps, 1), gun_dmg_per_fight: r(this.wd_gun),
        ability_weapon_dmg_per_fight: r(this.wd_kit), ability_melee_dmg_per_fight: r(this.md_kit),
        active_spirit_dmg_per_fight: r(this.sd_active), combat_s: T, shots_per_s: r(this.rps, 2),
        hits_per_fight: r(this.hits_ps * T, 1), headshots_per_fight: r(this.rate.headshot * T, 2), casts_per_fight: r(this.casts, 2) };
    }
  }

  class Engine {
    constructor(inputs, overrides) {
      this.I = inputs;
      this.A = Object.assign({}, inputs.A, overrides || {});
      this.props = inputs.rules.props;
      this.overrides = inputs.rules.items;
      this.items = inputs.items;
      this.byKey = new Map(this.items.map(it => [it.key, it]));
      this.heroes = new Map(inputs.heroes.map(h => [h.id, h]));
      this.avg = inputs.avg_hero;
    }
    ref(stage, hero) {
      return new Ref(this.A, stage, this.A.stages[stage], hero || this.avg, this.I.boon_souls, this.avg.melee, this.I.heroes);
    }
    _uses(cd, ref, trigger, perShotChance) {
      const T = ref.combat;
      if (perShotChance !== null && perShotChance !== undefined)
        return Math.min(ref.hits_ps * T * perShotChance, T / Math.max(truthy(cd) ? cd : 0.2, 0.2));
      if (trigger) {
        const lam = ref.rate[trigger];
        return (!truthy(cd) || cd <= 0) ? Math.min(1.0, lam * T) : renewal(cd, lam, T);
      }
      return (!truthy(cd) || cd <= 0) ? 1.0 : Math.min(60 / cd, Math.max(1.0, T / cd));
    }
    static _value(p, ref) {
      let v = p.v;
      if (v === null || v === undefined) return 0.0;
      if (p.st === "spirit") v += p.sv * ref.sp;
      else if (p.st === "power_increase") v += p.sv * ref.boons;
      return v;
    }
    _apply(ch, kind, q, ref, uses, up, dur, aoe, held) {
      const A = this.A;
      const add = (c, x) => { ch[c] = (ch[c] || 0) + x; };
      const tg = aoe ? A.targets : 1.0;
      const oocS = (60 - ref.combat) * A.ooc_efficiency;
      const span = () => (truthy(dur) ? Math.min(dur, ref.combat) * uses : ref.combat * up);
      switch (kind) {
        case "hp": add("ehp", (ref.ehp(null, ref.hp + q) - ref.ehp()) * up); break;
        case "hp_pct": add("ehp", (ref.ehp(null, ref.hp + ref.base_hp * q / 100) - ref.ehp()) * up); break;
        case "hp_pct_full": add("ehp", ref.ehp() * q / 100 * Math.min(1.0, uses)); break;
        case "resist_bullet": case "resist_spirit": case "resist_melee": {
          const t = kind.split("_")[1], r2 = Object.assign({}, ref.resist);
          r2[t] = 1 - (1 - r2[t]) * (1 - q / 100);
          add("ehp", (ref.ehp(r2) - ref.ehp()) * up); break;
        }
        case "barrier": add("ehp", q * uses); break;
        case "immune_bullet_s": add("ehp", ref.incoming_dps * ref.mix.bullet * q * Math.min(1.0, uses)); break;
        case "immune_all_s": add("ehp", ref.incoming_dps * q * Math.min(1.0, uses)); break;
        case "regen": add("heal", q * ref.combat * up); add("recovery", q * oocS * up); break;
        case "regen_proc": add("heal", q * ref.combat * up * tg); break;
        case "ooc_regen": add("recovery", q * oocS); break;
        case "ooc_regen_pct": add("recovery", q / 100 * ref.hp * oocS * 0.5); break;
        case "heal": add("heal", q * uses * tg); break;
        case "recovery_heal": add("recovery", q * Math.min(uses, 1.0)); break;
        case "heal_pct_hp": add("heal", q / 100 * ref.hp * uses); break;
        case "lifesteal_bullet": add("heal", q / 100 * ref.wd_shots * up); break;
        case "lifesteal_spirit": add("heal", q / 100 * ref.sd * up); break;
        case "heal_amp": add("heal", q / 100 * ref.heal_base * up); break;
        case "weapon_pct": add("weapon", q / 100 * ref.wd * up); break;
        case "fire_rate_pct": add("weapon", q / 100 * ref.wd_shots * up); break;
        case "self_damage_pct": add("weapon", q / 100 * ref.wd * up); add("spirit", q / 100 * ref.sd * up); break;
        case "clip_pct": add("clip", q * up); break;
        case "bullet_speed_pct": add("bullet_speed", q * up); break;
        case "bullet_proc": add("weapon", q * uses); break;
        case "bullet_proc_pct": add("weapon", q / 100 * ref.bullet * uses); break;
        case "melee_pct": add("melee", q / 100 * ref.md * up); break;
        case "spirit_power": add("spirit", q * ref.dsd_dsp * up); break;
        case "spirit_power_pct": add("spirit", q / 100 * ref.sp * ref.dsd_dsp * up); break;
        case "spirit_dmg": add("spirit", q * uses * tg); break;
        case "spirit_dps": add("spirit", q * span() * tg); break;
        case "spirit_pct_hp_dps": add("spirit", q / 100 * ref.hp * span() * tg); break;
        case "spirit_pct_hp": add("spirit", q / 100 * ref.hp * uses * tg); break;
        case "spirit_amp_proc": add("spirit", q / 100 * (ref.casts ? ref.sd_active / ref.casts : 0.0) * uses); break;
        case "cdr": {
          const f = (1 / (1 - Math.min(q, 90) / 100) - 1) * A.cdr_damage_share * up;
          add("spirit", ref.sd_active * f); add("weapon", ref.wd_active * f); break;
        }
        case "duration_pct": {
          const f = q / 100 * (1 - (ref.st.item_duration_pct || 0) / 100) * A.duration_damage_share * up;
          add("spirit", ref.sd_active * f); add("weapon", ref.wd_active * f); break;
        }
        case "enemy_bullet_resist_down": add("weapon", -q / 100 * ref.wd * (1 + A.team_amp) * up); break;
        case "enemy_spirit_resist_down": add("spirit", -q / 100 * ref.sd * (1 + A.team_amp) * up); break;
        case "enemy_damage_down": add("ehp", -q / 100 * ref.incoming_dps * ref.combat * up); break;
        case "enemy_spirit_damage_down": add("ehp", -q / 100 * ref.incoming_dps * ref.mix.spirit * ref.combat * up); break;
        case "enemy_fire_rate_down": add("ehp", q / 100 * ref.incoming_dps * ref.mix.bullet * ref.combat * up); break;
        case "reflect_bullet": case "reflect_spirit": {
          const t = kind === "reflect_bullet" ? "bullet" : "spirit";
          add("spirit", q / 100 * ref.incoming_dps * ref.mix[t] * span()); break;
        }
        case "self_drain": add("heal", -(q * ref.combat * up)); break;
        case "enemy_heal_down": add("heal_denied", -q / 100 * ref.enemy_heal * up); break;
        case "slow": add("slow", Math.abs(q) * span()); break;
        case "stun_s": case "silence_s": case "disarm_s": {
          const w = { stun_s: 1.0, silence_s: A.silence_weight, disarm_s: A.disarm_weight }[kind];
          add("hard_cc", q * uses * w * tg); break;
        }
        case "move": add("move", q * up); break;
        case "sprint": add("sprint", q * up); break;
        case "stamina": add("stamina", q * up); break;
        case "range_pct": add("range", q * (1 - (ref.st.item_range_pct || 0) / 100) * up); break;
        case "item_cdr": add("items", (ref.st.cd_item_value || 0) * (1 / (1 - Math.min(q, 90) / 100) - 1) * up); break;
        case "charges": add("charges", q); break;
        case "status_resist": add("status_resist", q * up); break;
        case "souls_per_min": add("souls", q * A.remaining_minutes); break;
        case "souls_held": add("souls", q * (truthy(held) ? held : A.remaining_minutes) * A.alive_share); break;
        case "echo": for (const [c, v] of Object.entries(ref.echo)) add(c, v * uses * q); break;
        case "refresh_ult": {
          const n = Math.min(uses, ref.ult_uses);
          for (const [c, v] of Object.entries(ref.ult)) add(c, v * n * A.ult_weight * q);
          break;
        }
        default: throw new Error("unknown kind " + kind);
      }
    }
    evaluate(key, ref) {
      const A = this.A;
      const it = this.byKey.get(key) || { blocks: [], item_cd: null };
      const ov = this.overrides[key] || {};
      const skip = new Set(ov.skip || []), remap = ov.remap || {}, trigger = ov.trigger;
      const held = isNum(ov.held_minutes) ? ov.held_minutes : null;
      const ch = {}, unmodeled = new Set();
      for (const b of it.blocks) {
        const btype = b.t, cd = b.cd, dur = b.dur, pval = b.pval;
        const chance = (ov.uses_per_min === "per_shot" && truthy(b.pc)) ? b.pc / 100 : null;
        let uses = this._uses(cd, ref, trigger, chance);
        if (ov.charge_up && trigger && truthy(cd)) uses = chargeUses(cd, ref.rate[trigger], ref.combat);
        if (isNum(ov.uses_per_min)) uses = ov.uses_per_min;
        const bu = pval.BuildUpPerShot;
        const upBuild = truthy(bu) ? Math.max(0.0, 1 - A.buildup_ref_s / bu / ref.combat) : null;
        for (const p of b.props) {
          const k = p.k;
          if (skip.has(k)) continue;
          const rule = Object.assign({}, this.props[k] || {});
          if (k in remap) rule.kind = remap[k];
          if (!Object.keys(rule).length) { unmodeled.add(k); continue; }
          const kind = rule.kind;
          if (kind === "ignore") continue;
          let q = Engine._value(p, ref) * (rule.factor !== undefined ? rule.factor : 1.0);
          if (q === 0) continue;
          const d = rule.dur_key ? (pval[rule.dur_key] !== undefined ? pval[rule.dur_key] : null) : dur;
          let up;
          if (btype === "Innate" && !truthy(cd)) up = p.c ? A.cond_uptime : 1.0;
          else if (truthy(cd) && !INSTANT.has(kind) && !OVER_DURATION.has(kind))
            up = Math.min(1.0, uses * (truthy(d) ? d : Math.min(cd, ref.combat) * 0.5) / ref.combat);
          else if (!truthy(cd) && (btype === "Passive" || btype === null || btype === undefined || btype === "Active"))
            up = (p.c || truthy(d) || btype === "Active") ? A.cond_uptime : 1.0;
          else up = 1.0;
          if (upBuild !== null && !INSTANT.has(kind)) up = upBuild;
          if (isNum(ov.uptime)) up = ov.uptime;
          let cm = 1.0;
          if (rule.cond === "low_hp") cm = A.low_hp_uptime / (up < 1 ? A.cond_uptime : 1);
          else if (rule.cond === "close") cm = A.close_share;
          else if (rule.cond === "long") cm = A.long_share;
          if (rule.per_stack) q *= b.stacks * A.stack_fill;
          this._apply(ch, kind, q, ref, uses * cm, up * cm, d, !!rule.aoe, held);
        }
      }
      for (const e of ov.add || []) {
        if (e.kind === "sell_refund") { ch.souls = (ch.souls || 0) + A.sell_refund * (it.cost || 0); continue; }
        const c = num(e.cooldown);
        const cd = truthy(c) ? c : it.item_cd;
        const use = e.use || "always";
        const uses = use !== "always" ? this._uses(cd, ref, e.trigger !== undefined ? e.trigger : trigger, null) : 1.0;
        const d = e.duration;
        const up = (use === "always" || !truthy(d)) ? 1.0 : Math.min(1.0, uses * d / ref.combat);
        this._apply(ch, e.kind, Number(e.value) * (e.factor !== undefined ? e.factor : 1.0), ref, uses, up, d, !!e.aoe, held);
      }
      if (held) {
        const f = held / A.item_horizon_minutes;
        for (const c of Object.keys(ch)) if (c !== "souls") ch[c] *= f;
      }
      const q = {}; for (const [k, v] of Object.entries(ch)) if (v) q[k] = v;
      return { q, unmodeled: [...unmodeled].sort(), note: ov.note || null };
    }
    prices(ref) {
      const A = this.A;
      const price = { souls: 1.0 };
      for (const [chan, key] of A.calibrate) {
        if (chan === "ehp") price.heal = null;
        const it = this.byKey.get(key);
        if (!it) continue;
        const { q } = this.evaluate(key, ref);
        let known = 0;
        for (const [c, v] of Object.entries(q)) if (truthy(price[c]) && c !== chan) known += v * price[c];
        const amount = q[chan] || 0, left = it.cost - known;
        if (amount > 0 && left > 0) { price[chan] = left / amount; if (chan === "ehp") price.heal = price.ehp; }
      }
      if (truthy(price.ehp)) {
        if (!(("hard_cc") in price)) price.hard_cc = A.stun_dps_multiple * ref.incoming_dps * price.ehp;
        if (!(("slow") in price)) price.slow = price.hard_cc * A.slow_to_stun / 100;
      }
      if (truthy(price.weapon) && truthy(price.spirit) && !("heal_denied" in price))
        price.heal_denied = (price.weapon + price.spirit) / 2;
      for (const [c, v] of Object.entries(A.fallback_prices || {})) if (!truthy(price[c])) price[c] = Number(v);
      const out = {}; for (const [c, v] of Object.entries(price)) if (truthy(v)) out[c] = v;
      return out;
    }
    run(heroId, pricing) {
      const hero = heroId ? this.heroes.get(Number(heroId)) : null;
      const rows = [], meta = { refs: {}, prices: {}, kit: {} };
      const refKeys = new Set(this.A.calibrate.map(([, k]) => k));
      const tierStage = {}; for (const [k, v] of Object.entries(this.A.stage_by_tier)) tierStage[Number(k)] = v;
      for (const sname of Object.keys(this.A.stages)) {
        const ref = this.ref(sname, hero);
        const price = this.prices(!hero || pricing === "own" ? ref : this.ref(sname, null));
        meta.refs[sname] = ref.describe(); meta.prices[sname] = price;
        if (ref.kit_rows) meta.kit[sname] = ref.kit_rows;
        for (const it of this.items) {
          const { q, unmodeled, note } = this.evaluate(it.key, ref);
          const row = { item_id: it.item_id, key: it.key, name: it.name, slot: it.slot, tier: it.tier, cost: it.cost,
            activation: it.activation, stage: sname, tier_stage: tierStage[it.tier] === sname };
          let value = 0, best = null, bestV = -Infinity;
          for (const c of CHANNELS) {
            const qc = q[c] || 0, vc = qc * (price[c] || 0);
            row["q_" + c] = qc; row["v_" + c] = vc; value += vc;
            if (vc > bestV) { bestV = vc; best = c; }
          }
          row.value = value; row.roi = it.cost ? value / it.cost : null;
          row.main_channel = bestV > 0 ? best : null;
          row.unmodeled = unmodeled.join(", "); row.n_unmodeled = unmodeled.length; row.note = note;
          row.reference = refKeys.has(it.key);
          rows.push(row);
        }
      }
      return { rows, meta };
    }
  }

  const IV = {
    CHANNELS, Engine, Ref,
    run(inputs, opts = {}) {
      return new Engine(inputs, opts.overrides).run(opts.hero || null, opts.pricing || "average");
    },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = IV;
  else root.IV = IV;
})(typeof globalThis !== "undefined" ? globalThis : this);
