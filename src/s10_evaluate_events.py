#!/usr/bin/env python3
"""S07 — Event-level evaluation. The headline result.

Produces, for the kickout-contest detector against your manual coding:
  * P / R / F1 at temporal IoU 0.3 / 0.5 / 0.7, each with a Wilson interval
  * PR curve and AP by sweeping the confidence threshold
  * mean and median absolute temporal offset |t_peak_pred - t_peak_gt|
  * bootstrap CI on F1, resampling EVENTS (the independent unit)

Everything is reported with n and an interval. With 12-20 events the
intervals are wide; that is the honest result and stating it plainly is
the point of the project.

Usage:
    python src/s07_evaluate_events.py --video-id gaa_lf_r3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir
from lib.logging_setup import get_logger
from lib.matching import match_events, temporal_iou
from lib.schema import read_table
from lib.statsx import prf, average_precision, bootstrap_ci


def f1_from_matches(m: np.ndarray) -> float:
    kinds = [r["kind"] for r in m]
    tp, fp, fn = kinds.count("TP"), kinds.count("FP"), kinds.count("FN")
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s07_eval_events", od / "logs")
    E = cfg["eval"]

    pred = read_table("events_pred", od / "events_pred.csv")
    gt = pd.read_csv(resolve(cfg, "gt") / args.video_id / "gt_events.csv")
    gt = gt[gt.coding_pass == 1]
    log.info("n_pred=%d  n_gt=%d", len(pred), len(gt))
    if len(gt) < 10:
        log.warning("fewer than 10 ground-truth events: every interval below will be very wide. "
                    "Do not report a point estimate without it.")

    out: dict = {"n_gt_events": int(len(gt)), "n_pred_events": int(len(pred)),
                 "by_tiou": {}, "operating_point": float(E.get("min_confidence", 0.0))}

    for tiou in E["tiou_thresholds"]:
        m = match_events(pred, gt, tiou)
        tp = int((m.kind == "TP").sum()); fp = int((m.kind == "FP").sum()); fn = int((m.kind == "FN").sum())
        res = prf(tp, fp, fn, E["alpha"])
        rec = m[m.kind == "TP"].to_dict("records") + m[m.kind == "FP"].to_dict("records") + \
              m[m.kind == "FN"].to_dict("records")
        pt, lo, hi = bootstrap_ci(rec, f1_from_matches,
                                  n_resamples=E["bootstrap"]["n_resamples"],
                                  ci=E["bootstrap"]["ci"], seed=cfg["project"]["seed"])
        out["by_tiou"][str(tiou)] = {
            "tp": tp, "fp": fp, "fn": fn,
            **{k: v.to_dict() for k, v in res.items()},
            "f1_bootstrap": {"value": pt, "ci_low": lo, "ci_high": hi,
                             "n_resamples": E["bootstrap"]["n_resamples"], "unit": "event"},
        }
        log.info("tIoU=%.2f | %s | %s | %s", tiou, res["precision"], res["recall"], res["f1"])
        if tiou == E["headline_tiou"]:
            m.to_csv(od / "event_matches.csv", index=False)

    # ---- PR curve over the confidence sweep -------------------------
    m = match_events(pred, gt, E["headline_tiou"]).query("kind != 'FN'")
    m = m.sort_values("confidence", ascending=False)
    ap_val, rec_curve, prec_curve = average_precision(m.confidence.values,
                                                      (m.kind == "TP").astype(int).values, n_gt=len(gt))
    out["average_precision"] = ap_val
    pd.DataFrame({"recall": rec_curve, "precision": prec_curve,
                  "confidence": m.confidence.values}).to_csv(od / "pr_curve_events.csv", index=False)

    # ---- temporal localisation -------------------------------------
    tps = match_events(pred, gt, E["headline_tiou"]).query("kind == 'TP'")
    if len(tps):
        off = (tps.t_peak_pred - tps.t_peak_gt).values
        out["temporal_offset_s"] = {
            "mean_abs": float(np.abs(off).mean()), "median_abs": float(np.median(np.abs(off))),
            "signed_mean": float(off.mean()),
            "p95_abs": float(np.percentile(np.abs(off), 95)), "n": int(len(off)),
        }
        log.info("mean |t_peak offset| = %.2fs (signed mean %+.2fs, n=%d) — a consistent sign "
                 "means a fixable systematic lag, scatter means an unfixable one",
                 out["temporal_offset_s"]["mean_abs"], out["temporal_offset_s"]["signed_mean"], len(off))

    (od / "metrics_events.json").write_text(json.dumps(out, indent=2, default=float))
    log.info("wrote metrics_events.json | AP=%.3f", ap_val)


if __name__ == "__main__":
    main()
