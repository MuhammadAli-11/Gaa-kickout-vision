#!/usr/bin/env python3
"""Extract continuous working segments around kickouts in an edited compilation.

How this differs from plan_segments.py
--------------------------------------
`plan_segments.py` picks a small number of FIXED-LENGTH windows out of a
full match, greedily covering as many kickouts as it can and accepting
that some fall outside. That is the right tool when the footage is
continuous and most of it is irrelevant.

A compilation is the opposite problem. Every kickout matters, they are
densely packed, and a fixed window would either miss some or emit the
same frames twice. So this tool pads each kickout individually, then
MERGES windows that overlap into one continuous segment. Each segment
keeps `timestamp_s = frame_idx / fps` exactly, and no frame is emitted
into two segments.

Overlapping vs touching
-----------------------
Windows that overlap share frames, so they must merge. Windows that
merely touch (one ends exactly where the next begins) share no frame.
`segments.merge_touching` decides, and it is a real decision rather than
a detail: it changes the segment count, and the segment is the unit that
train/val splits must respect, because frames inside one segment are
near-duplicates of each other.

The mapping table
-----------------
Every segment's clock restarts at zero. `<source>_segment_map.csv`
records the offset, so compilation time is recoverable as
`source_s = source_start_s + local_s`. This is written as a validated
table rather than left implicit in filenames, because an off-by-one
segment offset invalidates every temporal metric downstream and is
invisible when it happens.

Usage:
    python tools/extract_segments.py --source data/raw/data_2.mp4 \\
        --kickouts data/gt/data_2_kickouts.txt --source-id data_2 --dry-run
    python tools/extract_segments.py --source data/raw/data_2.mp4 \\
        --kickouts data/gt/data_2_kickouts.txt --source-id data_2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lib.config import load_config, resolve  # noqa: E402
from lib.logging_setup import get_logger  # noqa: E402
from lib.schema import write_table  # noqa: E402

log = get_logger("extract_segments")


def parse_timestamp(s: str) -> float | None:
    """Accept '2:35', '02:35', '1:02:35', or plain seconds."""
    s = s.strip()
    if not s or s.startswith("#"):
        return None
    s = re.sub(r"^\s*\d+\s*[.)]\s*", "", s)
    if " " in s:
        s = s.split(" ", 1)[0]
    if s.endswith(","):
        s = s[:-1]
    parts = s.split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    while len(vals) < 3:
        vals.insert(0, 0.0)
    return vals[0] * 3600 + vals[1] * 60 + vals[2]


def hhmmss(t: float) -> str:
    return f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{t % 60:06.3f}"


def probe(path: Path) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)]
    )
    meta = json.loads(out.decode("utf-8-sig"))
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return dict(
        duration_s=float(meta["format"]["duration"]),
        fps=float(num) / float(den) if float(den) else 0.0,
        width=int(v["width"]), height=int(v["height"]),
        codec=v.get("codec_name", "unknown"),
    )


def build_segments(times: list[float], pad_before: float, pad_after: float,
                   duration_s: float, merge_touching: bool) -> list[dict]:
    """Pad each kickout, then merge windows that share frames."""
    windows = [
        (max(t - pad_before, 0.0), min(t + pad_after, duration_s), i)
        for i, t in enumerate(times)
    ]
    windows.sort()

    segments: list[dict] = []
    for start, end, idx in windows:
        overlaps = segments and (
            start < segments[-1]["source_end_s"]
            or (merge_touching and start <= segments[-1]["source_end_s"])
        )
        if overlaps:
            seg = segments[-1]
            seg["source_end_s"] = max(seg["source_end_s"], end)
            seg["kickouts"].append(idx)
            seg["merged_windows"] += 1
        else:
            segments.append(dict(source_start_s=start, source_end_s=end,
                                 kickouts=[idx], merged_windows=1))
    return segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", required=True, help="the compilation video")
    ap.add_argument("--kickouts", required=True, help="one timestamp per line")
    ap.add_argument("--source-id", required=True, help="e.g. data_2")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write no video")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seg_cfg, vid_cfg = cfg["segments"], cfg["video"]
    src = Path(args.source)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    info = probe(src)
    log.info("source %s: %.1fs, %.3f fps, %dx%d, %s",
             src.name, info["duration_s"], info["fps"],
             info["width"], info["height"], info["codec"])

    if info["width"] < vid_cfg["width"] or info["height"] < vid_cfg["height"]:
        log.warning(
            "SOURCE IS SMALLER (%dx%d) THAN THE WORKING RESOLUTION (%dx%d). "
            "Rescaling up adds no information: a far-side player that is 14 px "
            "tall here is still 14 px of real detail after the upscale. Any "
            "small-object recall measured on this footage is a FLOOR, and it "
            "is not comparable to 720p source.",
            info["width"], info["height"], vid_cfg["width"], vid_cfg["height"])

    kick_path = Path(args.kickouts)
    if not kick_path.is_absolute():
        kick_path = ROOT / kick_path
    raw_lines = kick_path.read_text(encoding="utf-8").splitlines()
    times, labels = [], []
    for line in raw_lines:
        # Drop comments BEFORE splitting on commas. Splitting first lets prose
        # after a comma ("# 19 kickouts, as given") parse as a bare-seconds
        # timestamp, which silently adds a phantom kickout.
        line = line.split("#", 1)[0]
        if not line.strip():
            continue
        for tok in re.split(r"[,\n]", line):
            t = parse_timestamp(tok)
            if t is not None:
                times.append(t)
                labels.append(tok.strip())
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    labels = [labels[i] for i in order]
    log.info("%d kickout timestamps, %s to %s",
             len(times), hhmmss(times[0]), hhmmss(times[-1]))

    past = [t for t in times if t > info["duration_s"]]
    if past:
        raise SystemExit(f"{len(past)} timestamps beyond the end of the video")

    segments = build_segments(times, seg_cfg["pad_before_s"], seg_cfg["pad_after_s"],
                              info["duration_s"], bool(seg_cfg["merge_touching"]))

    naive = sum(min(t + seg_cfg["pad_after_s"], info["duration_s"])
                - max(t - seg_cfg["pad_before_s"], 0.0) for t in times)
    total = sum(s["source_end_s"] - s["source_start_s"] for s in segments)
    log.info("%d windows -> %d segments (%d merges)",
             len(times), len(segments), len(times) - len(segments))
    log.info("total %.1fs vs %.1fs unmerged — %.1fs of duplicate frames avoided",
             total, naive, naive - total)

    fps, W, H = int(vid_cfg["fps"]), int(vid_cfg["width"]), int(vid_cfg["height"])
    clips_dir = resolve(cfg, "clips")
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows, gt_rows = [], []
    for n, seg in enumerate(segments, 1):
        start, end = seg["source_start_s"], seg["source_end_s"]
        dur = end - start
        if dur < seg_cfg["min_segment_s"]:
            log.warning("segment %d is %.1fs, below min_segment_s — skipped", n, dur)
            continue
        video_id = f"{args.source_id}_seg{n:02d}"
        n_frames = int(round(dur * fps))

        log.info("seg%02d  %s -> %s  (%.1fs, %d frames, %d kickout%s)",
                 n, hhmmss(start), hhmmss(end), dur, n_frames,
                 len(seg["kickouts"]), "" if len(seg["kickouts"]) == 1 else "s")

        for k, gi in enumerate(seg["kickouts"], 1):
            local = times[gi] - start
            gt_rows.append(dict(
                video_id=video_id, event_id=f"ko_{k:03d}",
                source_timestamp=labels[gi], source_s=times[gi],
                t_peak_s_approx=round(local, 2),
                t_start_s="", t_end_s="", t_peak_s="",
                n_players_in_contest="", outcome="", restart_type="",
                coder_id="", coding_pass=1, visibility="", notes="",
            ))

        if not args.dry_run:
            clip = clips_dir / f"{video_id}_working.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
                 "-t", f"{dur:.3f}",
                 "-vf", f"fps={fps},scale={W}:{H}:flags=lanczos",
                 "-fps_mode", "cfr", "-pix_fmt", vid_cfg["pix_fmt"],
                 "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an",
                 str(clip)], check=True)

        rows.append(dict(
            segment_id=n, video_id=video_id, source_id=args.source_id,
            source_start_s=round(start, 3), source_end_s=round(end, 3),
            duration_s=round(dur, 3), n_frames=n_frames, fps=fps,
            width=W, height=H, n_kickouts=len(seg["kickouts"]),
            merged_windows=seg["merged_windows"],
            src_width=info["width"], src_height=info["height"],
        ))

    gt_dir = resolve(cfg, "gt")
    map_path = gt_dir / f"{args.source_id}_segment_map.csv"
    if args.dry_run:
        log.info("dry run — no video and no tables written")
        print("\n" + pd.DataFrame(rows)[
            ["video_id", "source_start_s", "source_end_s", "duration_s",
             "n_frames", "n_kickouts", "merged_windows"]].to_string(index=False))
        return

    write_table(pd.DataFrame(rows), "segment_map", map_path)
    log.info("wrote %s", map_path)

    for vid, g in pd.DataFrame(gt_rows).groupby("video_id"):
        d = gt_dir / str(vid)
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["video_id"]).to_csv(d / "gt_events_skeleton.csv", index=False)
    log.info("wrote %d per-segment kickout skeletons", len(rows))

    log.info("RECOVER COMPILATION TIME: source_s = source_start_s + local_s "
             "(the map is the only place that offset is recorded)")
    log.info("t_peak_s_approx is your noted timestamp converted, NOT ground "
             "truth — step frame by frame before coding from it")


if __name__ == "__main__":
    main()
