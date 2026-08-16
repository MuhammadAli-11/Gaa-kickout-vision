#!/usr/bin/env python3
"""S03 — Multi-object tracking (ByteTrack) to give players persistent IDs.

Expect this to be the weakest link, and treat that as the finding.
Broadcast direction cuts and pans at exactly the moment of the contest,
so identity is lost when it is most needed. s07/s09 quantify it; do not
tune it away by shrinking the clip to static passages.

Usage:
    python src/s03_track.py --video-id gaa_lf_r3
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir, set_seeds, make_provenance
from lib.logging_setup import get_logger
from lib.schema import write_table
from lib.timebase import frame_to_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg)
    od = out_dir(cfg)
    log = get_logger("s03_track", od / "logs")
    t = cfg["track"]

    from ultralytics import YOLO
    model = YOLO(cfg["detect"]["weights"])
    clip = resolve(cfg, "clips") / f"{args.video_id}_working.mp4"

    t0 = time.perf_counter()
    rows = []
    for fi, r in enumerate(model.track(
        source=str(clip), stream=True, persist=True,
        tracker=f"{t['tracker']}.yaml",
        imgsz=cfg["detect"]["imgsz"], conf=cfg["detect"]["conf"], iou=cfg["detect"]["iou"],
        classes=cfg["detect"]["classes"], max_det=cfg["detect"]["max_det"], verbose=False,
    )):
        b = r.boxes
        if b is None or b.id is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        for k in range(len(ids)):
            x1, y1, x2, y2 = xyxy[k]
            rows.append((ids[k], fi, 0.0, x1, y1, x2, y2,
                         (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1, (x2 - x1) * (y2 - y1), conf[k]))
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows, columns=["track_id", "frame_idx", "timestamp_s", "x1", "y1", "x2", "y2",
                                     "cx", "cy", "bbox_h", "bbox_area", "conf"])
    df["timestamp_s"] = frame_to_time(df["frame_idx"], cfg["video"]["fps"])

    # Drop flicker tracks: they add noise to every scene feature.
    lens = df.groupby("track_id").size()
    keep = lens[lens >= t["min_track_len_frames"]].index
    dropped = df.track_id.nunique() - len(keep)
    df = df[df.track_id.isin(keep)].copy()

    df = df.astype({c: "float32" for c in ["x1", "y1", "x2", "y2", "cx", "cy", "bbox_h", "bbox_area", "conf"]})
    write_table(df, "tracks", od / "tracks.parquet")

    per_frame = df.groupby("frame_idx").size()
    log.info("tracks=%d (dropped %d shorter than %d frames) rows=%d wall=%.1fs",
             df.track_id.nunique(), dropped, t["min_track_len_frames"], len(df), elapsed)
    log.info("mean tracks/frame=%.1f  median track length=%.1f frames (%.2fs)",
             per_frame.mean(), np.median(lens[keep]), np.median(lens[keep]) / cfg["video"]["fps"])
    log.info("RED FLAG CHECK: if unique track_ids ~= mean tracks/frame, tracking is not "
             "actually associating across frames.")

    make_provenance("s03_track", cfg, [str(clip)], dict(**t, wall_s=elapsed)).write(
        od / "provenance_s03.json")


if __name__ == "__main__":
    main()
