#!/usr/bin/env python3
"""S05 — Candidate kickout-contest detection from the feature series.

Two modes:
  rule    : transparent, no training data, no leakage risk. Do this first.
  learned : logistic regression / GBDT on windowed statistics, with
            TEMPORAL BLOCK cross-validation and a purge gap. Random
            splits leak, because adjacent 3 s windows share frames.
            With ~15 positives, treat the learned mode as an
            illustration of correct methodology, not as a result.

Usage:
    python src/s05_detect_events.py --video-id gaa_lf_r3 [--mode rule|learned]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lib.config import load_config, out_dir, set_seeds, make_provenance
from lib.logging_setup import get_logger
from lib.schema import read_table, write_table

WINDOW_STATS = ["mean", "std", "min", "max"]
FEATURE_COLS = ["n_tracks", "cluster_density", "max_bbox_height", "centroid_spread",
                "mean_vertical_velocity", "max_vertical_velocity"]


def windows(feat: pd.DataFrame, fps: int, window_s: float, stride_s: float) -> pd.DataFrame:
    w = int(round(window_s * fps)); s = int(round(stride_s * fps))
    rows = []
    for start in range(0, max(len(feat) - w, 0) + 1, s):
        chunk = feat.iloc[start:start + w]
        r = {"w_start_s": chunk.timestamp_s.iloc[0], "w_end_s": chunk.timestamp_s.iloc[-1]}
        for c in FEATURE_COLS:
            v = chunk[c].astype(float)
            r[f"{c}_mean"] = v.mean(); r[f"{c}_std"] = v.std(ddof=0)
            r[f"{c}_min"] = v.min(); r[f"{c}_max"] = v.max()
        r["cut_frac"] = chunk.scene_change_flag.mean()
        r["valid_frac"] = chunk.valid_frame.mean()
        r["t_peak_s"] = float(chunk.loc[chunk.mean_vertical_velocity.idxmax(), "timestamp_s"]) \
            if chunk.mean_vertical_velocity.notna().any() else float(chunk.timestamp_s.mean())
        r["n_players_in_contest"] = int(chunk.n_tracks.max())
        rows.append(r)
    return pd.DataFrame(rows)


def score_rule(W: pd.DataFrame, cfg: dict) -> np.ndarray:
    r = cfg["events"]["rule"]
    spread_thr = np.nanpercentile(W.centroid_spread_mean, r["spread_percentile"])
    vel_thr = np.nanpercentile(W.mean_vertical_velocity_max, r["velocity_percentile"])
    # Each term in [0,1]; product keeps the score interpretable and makes
    # the confidence axis of the PR curve meaningful rather than arbitrary.
    compress = np.clip((spread_thr - W.centroid_spread_mean) / max(spread_thr, 1e-6) + 0.5, 0, 1)
    leap = np.clip(W.mean_vertical_velocity_max / max(vel_thr, 1e-6), 0, 1)
    density = np.clip(W.n_tracks_max / max(r["min_tracks"], 1), 0, 1)
    quality = np.clip(1.0 - W.cut_frac, 0, 1) * W.valid_frac
    return (compress * leap * density * quality).fillna(0).values


def nms_time(W: pd.DataFrame, scores: np.ndarray, min_gap_s: float) -> list[int]:
    keep, order = [], np.argsort(-scores)
    for i in order:
        if all(abs(W.t_peak_s.iloc[i] - W.t_peak_s.iloc[j]) >= min_gap_s for j in keep):
            keep.append(int(i))
    return sorted(keep, key=lambda i: W.t_peak_s.iloc[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--mode", default=None, choices=["rule", "learned"])
    ap.add_argument("--min-confidence", type=float, default=0.05,
                    help="keep low: s07 sweeps this to build the PR curve")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg)
    od = out_dir(cfg)
    log = get_logger("s05_events", od / "logs")
    ev = cfg["events"]
    mode = args.mode or ev["mode"]

    feat = read_table("features", od / "features.parquet")
    W = windows(feat, cfg["video"]["fps"], ev["window_s"], ev["stride_s"])
    log.info("%d candidate windows (%.1fs window, %.1fs stride)", len(W), ev["window_s"], ev["stride_s"])

    if mode == "rule":
        scores = score_rule(W, cfg)
    else:
        from sklearn.linear_model import LogisticRegression
        from lib.cv_blocks import temporal_block_cv, label_windows
        gt = pd.read_csv(od.parent.parent / "data" / "gt" / args.video_id / "gt_events.csv")
        y = label_windows(W, gt, tiou_thresh=0.5)
        X = W[[c for c in W.columns if any(c.endswith(f"_{s}") for s in WINDOW_STATS)]].fillna(0).values
        scores = temporal_block_cv(X, y, W.w_start_s.values, LogisticRegression(max_iter=2000, class_weight="balanced"),
                                   n_splits=ev["learned"]["n_splits"], gap_s=ev["learned"]["block_gap_s"])
        log.info("learned mode: %d positive windows of %d — treat coefficients as descriptive only",
                 int(y.sum()), len(y))

    keep = nms_time(W, scores, ev["rule"]["min_gap_s"])
    keep = [i for i in keep if scores[i] >= args.min_confidence]

    out = pd.DataFrame([dict(
        event_id=f"pred_{n:03d}",
        t_start_s=float(W.w_start_s.iloc[i]), t_end_s=float(W.w_end_s.iloc[i]),
        t_peak_s=float(W.t_peak_s.iloc[i]),
        duration_s=float(W.w_end_s.iloc[i] - W.w_start_s.iloc[i]),
        confidence=float(scores[i]), n_players_in_contest=int(W.n_players_in_contest.iloc[i]),
        detector_mode=mode,
    ) for n, i in enumerate(keep, 1)])

    write_table(out, "events_pred", od / "events_pred.csv")
    log.info("%d candidate events kept after temporal NMS (min gap %.1fs)", len(out), ev["rule"]["min_gap_s"])
    log.info("SANITY: a 10-minute passage should contain roughly 12-20 kickouts. "
             "Hundreds of candidates means the score is not discriminating; "
             "two means the thresholds are strangling recall.")
    make_provenance("s05_events", cfg, ["features.parquet"], {"resolved_mode": mode, **ev}).write(
        od / "provenance_s05.json")


if __name__ == "__main__":
    main()
