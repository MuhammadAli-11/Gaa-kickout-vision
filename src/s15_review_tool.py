#!/usr/bin/env python3
"""S12 — Human-factors sub-study: analyst-in-the-loop review and timing.

OPTIONAL, and the reason it exists: the studentship is explicitly about
Human Factors and Ergonomics in sport performance intelligence, not only
about computer vision. A CV project that also measures how a human
interacts with the model's output is answering the actual advertised
research question rather than half of it.

Protocol (docs/10):
  Condition A  manual_from_scratch  — code kickouts with no model output
  Condition B  model_assisted       — accept/reject/adjust model candidates

Records per decision: model confidence, verdict, seconds to judge. That
yields three genuinely interesting quantities from one afternoon:
  1. time per event coded, A vs B  (does the tool actually save time?)
  2. accept rate by confidence band (is the confidence score TRUSTWORTHY?)
  3. automation bias: acceptance of model errors under condition B

Usage:
    python src/s12_review_tool.py --video-id gaa_lf_r3 --condition model_assisted
"""
from __future__ import annotations

import argparse
import time
import uuid

import pandas as pd

from lib.config import load_config, out_dir
from lib.logging_setup import get_logger
from lib.schema import read_table, write_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--condition", default="model_assisted",
                    choices=["manual_from_scratch", "model_assisted"])
    ap.add_argument("--blind", action="store_true",
                    help="hide the model confidence from the reviewer (control for anchoring)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s12_review", od / "logs")
    pred = read_table("events_pred", od / "events_pred.csv").sort_values("t_peak_s")

    print(f"\nReview: {len(pred)} candidates. Open the clip alongside this terminal.")
    print("For each candidate: [a]ccept  [r]eject  [j] adjust t_peak  [q]uit\n")

    rows = []
    for r in pred.itertuples():
        conf = "" if args.blind else f"  conf={r.confidence:.2f}"
        t0 = time.perf_counter()
        ans = input(f"{r.event_id}  t={r.t_peak_s:7.2f}s{conf}  > ").strip().lower()
        dt = time.perf_counter() - t0
        if ans.startswith("q"):
            break
        verdict = {"a": "accept", "r": "reject", "j": "adjust"}.get(ans[:1], "reject")
        adj = float(input("   corrected t_peak (s): ")) if verdict == "adjust" else float("nan")
        rows.append(dict(review_id=uuid.uuid4().hex[:8], event_id=r.event_id,
                         model_confidence=float(r.confidence), analyst_verdict=verdict,
                         seconds_to_judge=dt, adjusted_t_peak_s=adj, condition=args.condition))

    df = pd.DataFrame(rows)
    path = od / "review_log.csv"
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    write_table(df, "review_log", path)

    if len(rows):
        d = pd.DataFrame(rows)
        log.info("median time per decision: %.1fs | accept rate: %.0f%%",
                 d.seconds_to_judge.median(), 100 * (d.analyst_verdict == "accept").mean())
        bands = cfg["human_factors"]["confidence_bands"]
        d["band"] = pd.cut(d.model_confidence, bands)
        log.info("accept rate by confidence band:\n%s",
                 d.groupby("band", observed=False).analyst_verdict.apply(lambda s: (s == "accept").mean()))
        log.info("Calibration question for the write-up: does acceptance rise monotonically with "
                 "confidence? If not, the confidence score should not be shown to an analyst.")


if __name__ == "__main__":
    main()
