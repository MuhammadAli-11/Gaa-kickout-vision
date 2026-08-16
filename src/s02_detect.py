#!/usr/bin/env python3
"""S02 — Person detection with YOLO over the working clip.

Notes that matter for the write-up:
  * imgsz stays at 1280. Dropping to 640 halves the pixel height of a
    far-side player and is the single biggest cause of the small-object
    recall collapse you will report in §6.1.
  * conf starts LOW (0.25). Recall is the scarce resource here; the
    tracker's low-threshold association stage can use weak detections.
  * Detections are written raw and unfiltered. All thresholding happens
    downstream so that a PR sweep is possible without re-running inference.

Usage:
    python src/s02_detect.py --video-id gaa_lf_r3 [--source clip|frames]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from lib.config import load_config, resolve, out_dir, set_seeds, make_provenance
from lib.logging_setup import get_logger
from lib.schema import write_table
from lib.timebase import frame_to_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--source", default="clip", choices=["clip", "frames"])
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on first N frames")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seeds(cfg)
    od = out_dir(cfg)
    log = get_logger("s02_detect", od / "logs")
    d = cfg["detect"]

    from ultralytics import YOLO
    model = YOLO(d["weights"])

    src = (resolve(cfg, "clips") / f"{args.video_id}_working.mp4" if args.source == "clip"
           else resolve(cfg, "frames") / args.video_id)

    log.info("detector=%s imgsz=%d conf=%.2f device=%s", d["weights"], d["imgsz"], d["conf"], d["device"])
    t0 = time.perf_counter()
    results = model.predict(
        source=str(src), stream=True, imgsz=d["imgsz"], conf=d["conf"], iou=d["iou"],
        classes=d["classes"], max_det=d["max_det"], verbose=False,
        device=None if d["device"] == "auto" else d["device"],
        half=bool(d["half"]),
    )

    rows = []
    n_frames = 0
    for fi, r in enumerate(results):
        if args.limit and fi >= args.limit:
            break
        n_frames += 1
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        order = conf.argsort()[::-1]
        for di, k in enumerate(order):
            rows.append((fi, di, 0.0, *xyxy[k], conf[k], cls[k]))
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows, columns=["frame_idx", "det_idx", "timestamp_s",
                                     "x1", "y1", "x2", "y2", "conf", "class_id"])
    df["timestamp_s"] = frame_to_time(df["frame_idx"], cfg["video"]["fps"])
    df = df.astype({"frame_idx": "int64", "det_idx": "int64", "x1": "float32", "y1": "float32",
                    "x2": "float32", "y2": "float32", "conf": "float32", "class_id": "int16"})
    write_table(df, "detections", od / "detections.parquet")

    fps_proc = n_frames / elapsed if elapsed else 0.0
    log.info("frames=%d dets=%d wall=%.1fs throughput=%.2f fps (%.1f GPU-s per video-minute)",
             n_frames, len(df), elapsed, fps_proc,
             elapsed / max(n_frames / cfg["video"]["fps"] / 60, 1e-9))
    log.info("mean detections/frame = %.2f (expect 8-22 on a wide GAA shot)",
             len(df) / max(n_frames, 1))

    make_provenance("s02_detect", cfg, [str(src)],
                    dict(**d, n_frames=n_frames, wall_s=elapsed, throughput_fps=fps_proc)
                    ).write(od / "provenance_s02.json")


if __name__ == "__main__":
    main()
