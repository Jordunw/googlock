"""Run the analysis stages over data/clean and write data/results.

    python -m dla.analyze                 # all stages
    python -m dla.analyze --only heroes   # comma-separated subset: heroes,items,counters,output,fits,value
                                          # (+ output_enemy: just the per-enemy part of output)

Stage order matters: items needs hero_strength; counters and output reuse the purchase
scoring from items (same session); fits needs hero_item_effects. value (item value from
game data) needs no match data.
"""
from __future__ import annotations

import argparse
import logging
import time

from . import itemvalue
from .analysis import counters, fits, heroes, items, output
from .analysis.common import Store
from .config import load_settings

STAGES = ["heroes", "items", "counters", "output", "fits", "value"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings")
    ap.add_argument("--only", help="comma-separated stages")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("dla.analyze")
    st = Store(load_settings(a.settings))
    todo = a.only.split(",") if a.only else STAGES
    scored = False
    for stage in STAGES[:4] + ["output_enemy"] + STAGES[4:]:
        if stage not in todo:
            continue
        t = time.time()
        log.info("=== %s ===", stage)
        if stage == "heroes":
            heroes.run(st)
        elif stage == "items":
            items.run(st)
            scored = True
        elif stage in ("counters", "output"):
            if not scored:   # both need the scored purchases (P(win) at purchase)
                log.info("scoring purchases for %s", stage)
                items.run(st)
                scored = True
            (counters if stage == "counters" else output).run(st)
        elif stage == "output_enemy":   # per-enemy part of output (already included in "output")
            output.run_enemy(st)
        elif stage == "fits":
            fits.run(st)
        elif stage == "value":
            itemvalue.run(st.s)
        log.info("%s done in %.0fs", stage, time.time() - t)


if __name__ == "__main__":
    main()
