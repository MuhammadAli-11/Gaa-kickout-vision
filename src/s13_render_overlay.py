#!/usr/bin/env python3
"""S10 — Overlay video: boxes, track IDs, event banner, and a labelled failure.

The highlight MUST contain at least one error, annotated as an error on
screen. A demo reel of successes tells a supervisor nothing about whether
you can evaluate your own work; a reel that says "this one is a false
positive, here is why" tells them everything.

Usage:
    python src/s10_render_overlay.py --video-id gaa_lf_r3 --segments auto
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir
from lib.logging_setup import get_logger
from lib.schema import read_table

GREEN, RED, AMBER, WHITE = (80, 220, 100), (60, 60, 235), (0, 190, 255), (255, 255, 255)


def pick_segments(matches: pd.DataFrame, dur_s: float, pad: float = 6.0) -> list[tuple[float, float, str]]:
    """Choose 3-4 windows: at least one TP, and at least one FP or FN."""
    segs = []
    tps = matches[matches.kind == "TP"].nlargest(2, "confidence")
    errs = pd.concat([matches[matches.kind == "FP"].nlargest(1, "confidence"),
                      matches[matches.kind == "FN"].head(1)])
    for _, r in pd.concat([tps, errs]).iterrows():
        t = r.t_peak_pred if pd.notna(r.t_peak_pred) else r.t_peak_gt
        segs.append((max(t - pad, 0), min(t + pad, dur_s), r.kind))
    return sorted(segs)[:4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s10_render", od / "logs")
    fps = cfg["video"]["fps"]

    tracks = read_table("tracks", od / "tracks.parquet")
    pred = read_table("events_pred", od / "events_pred.csv")
    gt = pd.read_csv(resolve(cfg, "gt") / args.video_id / "gt_events.csv").query("coding_pass == 1")
    matches = pd.read_csv(od / "event_matches.csv")

    cap = cv2.VideoCapture(str(resolve(cfg, "clips") / f"{args.video_id}_working.mp4"))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segs = pick_segments(matches, n_frames / fps)
    if cfg["render"]["must_include_a_failure"] and not any(s[2] in ("FP", "FN") for s in segs):
        log.warning("no error segment selected — the demo loses most of its value. "
                    "Check that s07 wrote event_matches.csv with FP/FN rows.")
    log.info("segments: %s", [(round(a, 1), round(b, 1), k) for a, b, k in segs])

    out = cv2.VideoWriter(str(od / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (cfg["video"]["width"], cfg["video"]["height"]))
    by_frame = {f: g for f, g in tracks.groupby("frame_idx")}

    for a, b, kind in segs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps))
        for fi in range(int(a * fps), int(b * fps)):
            ok, img = cap.read()
            if not ok:
                break
            t = fi / fps
            for r in by_frame.get(fi, pd.DataFrame()).itertuples():
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), GREEN, 1)
                if cfg["render"]["draw_track_ids"]:
                    cv2.putText(img, str(int(r.track_id)), (int(r.x1), int(r.y1) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)
            in_pred = ((pred.t_start_s <= t) & (t <= pred.t_end_s)).any()
            in_gt = ((gt.t_start_s <= t) & (t <= gt.t_end_s)).any()
            label, colour = {
                (True, True): ("PREDICTED CONTEST - correct", GREEN),
                (True, False): ("PREDICTED CONTEST - FALSE POSITIVE", RED),
                (False, True): ("MISSED CONTEST - FALSE NEGATIVE", AMBER),
                (False, False): ("", WHITE),
            }[(bool(in_pred), bool(in_gt))]
            if label:
                cv2.rectangle(img, (0, 0), (cfg["video"]["width"], 46), (25, 25, 25), -1)
                cv2.putText(img, label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
            cv2.putText(img, f"t={t:6.2f}s  tracks={len(by_frame.get(fi, []))}",
                        (14, cfg["video"]["height"] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            out.write(img)

    out.release(); cap.release()
    log.info("wrote %s", od / "overlay.mp4")


if __name__ == "__main__":
    main()
