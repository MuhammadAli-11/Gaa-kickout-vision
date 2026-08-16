#!/usr/bin/env python3
"""S01 — Standardisation. Trim, fix fps/resolution, extract frames, build manifest.

The whole project's time axis is defined here. After this stage:
  * constant frame rate, 0-based contiguous frame indices
  * timestamp_s == frame_idx / fps, exactly
Any timing metric computed off a variable-frame-rate broadcast file is
wrong in a way that is invisible until someone checks. This is the
stage to be able to talk about in the interview.

Usage:
    python src/s01_prepare_footage.py --video-id gaa_lf_r3
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from lib.config import load_config, resolve, make_provenance
from lib.logging_setup import get_logger
from lib.schema import write_table
from lib.timebase import frame_to_time


def run(cmd: list[str], log) -> None:
    log.info("$ %s", " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--skip-frames", action="store_true",
                    help="produce the clip only (Ultralytics can read video directly)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    v = dict(cfg["video"])  # copy: per-window trim must not leak into cfg/run_id
    log = get_logger("s01_prepare")

    reg = pd.read_csv(resolve(cfg, "raw") / "video_registry.csv")
    if args.video_id not in set(reg.video_id):
        raise SystemExit(f"{args.video_id} not in registry — run s00_ingest.py first")
    src = resolve(cfg, "raw") / reg.loc[reg.video_id == args.video_id, "filename"].iloc[0]

    # The segment plan is the source of truth for where each working window
    # starts. Reading it here means the window cannot silently drift from the
    # plan because a config edit was forgotten between runs. Config remains the
    # fallback for any video_id the plan does not list.
    plan_path = resolve(cfg, "gt") / "segment_plan.csv"
    trim_from = args.config
    if plan_path.exists():
        plan = pd.read_csv(plan_path)
        hit = plan.loc[plan.video_id == args.video_id]
        if len(hit):
            v["segment_start"] = str(hit.segment_start.iloc[0])
            v["segment_duration"] = str(hit.segment_duration.iloc[0])
            trim_from = plan_path.name
    log.info("trim %s +%s (from %s)", v["segment_start"], v["segment_duration"], trim_from)

    clip = resolve(cfg, "clips") / f"{args.video_id}_working.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)

    # -ss BEFORE -i for fast seek; re-encode (not -c copy) so the trim is
    # frame-accurate and the output genuinely has constant frame rate.
    run(["ffmpeg", "-y", "-ss", v["segment_start"], "-i", str(src),
         "-t", v["segment_duration"],
         "-vf", f"fps={v['fps']},scale={v['width']}:{v['height']}:flags=lanczos",
         "-fps_mode", "cfr", "-pix_fmt", v["pix_fmt"],  # -vsync: removed in ffmpeg 8+
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", str(clip)], log)

    frames_dir = resolve(cfg, "frames") / args.video_id
    if not args.skip_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-y", "-i", str(clip), "-q:v", str(v["jpeg_quality"]),
             "-start_number", "0", str(frames_dir / "%06d.jpg")], log)

    n = len(sorted(frames_dir.glob("*.jpg"))) if not args.skip_frames else int(
        subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                                 "-of", "csv=p=0", str(clip)], text=True).strip())

    idx = list(range(n))
    man = pd.DataFrame(dict(
        frame_idx=idx,
        timestamp_s=frame_to_time(idx, v["fps"]),
        path=[str(frames_dir / f"{i:06d}.jpg") for i in idx],
        width=v["width"], height=v["height"], video_id=args.video_id,
    ))
    out = resolve(cfg, "interim") / args.video_id / "manifest.csv"
    write_table(man, "frame_manifest", out)

    log.info("clip=%s frames=%d duration=%.2fs", clip.name, n, n / v["fps"])
    log.info("SANITY: frame %d -> %.3fs; last frame -> %.3fs", 0, 0.0, (n - 1) / v["fps"])
    make_provenance("s01_prepare", cfg, [str(src)], dict(n_frames=n, **v)).write(
        resolve(cfg, "interim") / args.video_id / "provenance_s01.json")


if __name__ == "__main__":
    main()
