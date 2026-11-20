"""Run the analysis stages over data/clean and write data/results.

    python -m dla.analyze                 # all stages
    python -m dla.analyze --only heroes   # comma-separated subset: heroes,items,counters,fits

Stage order matters: items needs hero_strength; counters reuses the purchase scoring from
items (same session); fits needs hero_item_effects.
"""
from __future__ import annotations

import argparse
import logging
import time

from .analysis import counters, fits, heroes, items
from .analysis.common import Store
from .config import load_settings

STAGES = ["heroes", "items", "counters", "fits"]


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
    for stage in STAGES:
        if stage not in todo:
            continue
        t = time.time()
        log.info("=== %s ===", stage)
        if stage == "heroes":
            heroes.run(st)
        elif stage == "items":
            items.run(st)
            scored = True
        elif stage == "counters":
            if not scored:   # counters' win signal needs the scored purchases
                log.info("scoring purchases for counters")
                items.run(st)
            counters.run(st)
        elif stage == "fits":
            fits.run(st)
        log.info("%s done in %.0fs", stage, time.time() - t)


if __name__ == "__main__":
    main()
