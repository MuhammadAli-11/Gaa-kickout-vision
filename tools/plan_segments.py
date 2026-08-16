#!/usr/bin/env python3
"""Plan working segments from a list of kickout timestamps.

The problem this solves
-----------------------
Your kickouts are spread across a 55-minute match, but the pipeline works
on a short continuous clip whose time axis starts at zero. Two bad
options present themselves: process the whole match (slow, and most of it
is irrelevant), or cut 25 tiny clips around each kickout (fast, but
destroys the between-kickout footage where false positives live, and
gives you 25 separate time origins to reconcile).

This picks a middle path: a small number of CONTINUOUS windows that
between them cover as many kickouts as possible. Each window becomes its
own video_id and its own clip, so within a window `timestamp_s =
frame_idx / fps` still holds exactly.

It also does the arithmetic that would otherwise bite you: converting
your source-time kickout list into window-local seconds. Getting that
offset wrong by one window shifts every temporal metric silently, and it
is the single easiest way to invalidate a weekend's work.

Usage:
    python tools/plan_segments.py --kickouts data/gt/kickouts_source.txt \\
        --window-minutes 12 --max-windows 2 --video-id lgf26_final
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_timestamp(s: str) -> float | None:
    """Accept '2:35', '02:35', '1:02:35', or plain seconds."""
    s = s.strip()
    if not s or s.startswith("#"):
        return None
    s = re.sub(r"^\s*\d+\s*[.)]\s*", "", s)          # strip list numbering
    note = ""
    if " " in s:
        s, note = s.split(" ", 1)
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


def pick_windows(times: list[float], window_s: float, max_windows: int,
                 pad_s: float = 20.0) -> list[tuple[float, float, list[int]]]:
    """Greedy: repeatedly take the window covering the most uncovered kickouts."""
    remaining = set(range(len(times)))
    windows = []
    for _ in range(max_windows):
        if not remaining:
            break
        best = None
        for i in sorted(remaining):
            start = max(times[i] - pad_s, 0.0)
            end = start + window_s
            covered = [j for j in remaining if start + pad_s <= times[j] <= end - pad_s]
            if best is None or len(covered) > len(best[2]):
                best = (start, end, covered)
        if not best or not best[2]:
            break
        windows.append(best)
        remaining -= set(best[2])
    return sorted(windows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kickouts", required=True, help="text file, one timestamp per line")
    ap.add_argument("--video-id", default="lgf26_final")
    ap.add_argument("--window-minutes", type=float, default=12.0)
    ap.add_argument("--max-windows", type=int, default=2)
    ap.add_argument("--pad-seconds", type=float, default=20.0,
                    help="keep this much footage before/after the outermost kickout")
    ap.add_argument("--min-gap-s", type=float, default=25.0,
                    help="below this, two entries are flagged as a probable replay")
    args = ap.parse_args()

    raw = Path(args.kickouts).read_text().splitlines()
    parsed = [(ln, parse_timestamp(ln)) for ln in raw]
    times = [t for _, t in parsed if t is not None]
    labels = [ln.strip() for ln, t in parsed if t is not None]
    times, labels = zip(*sorted(zip(times, labels))) if times else ([], [])
    times, labels = list(times), list(labels)
    print(f"parsed {len(times)} timestamps\n")

    # ---- flag entries that cannot both be real kickouts --------------
    suspicious = []
    for i in range(1, len(times)):
        gap = times[i] - times[i - 1]
        if gap < args.min_gap_s:
            suspicious.append((labels[i - 1], labels[i], gap))
    if suspicious:
        print("!! CHECK THESE — too close together to both be kickouts:")
        for a, b, g in suspicious:
            print(f"   {a}  and  {b}   ({g:.0f} s apart)")
        print("   A kickout cannot follow another in under ~25 s: the ball has to go dead,")
        print("   players reset, the keeper restarts. These are almost certainly REPLAYS,")
        print("   or one timestamp is the replay of the other. Re-watch, decide, and log")
        print("   the replay in annotation_notes.md — replays are the source paper's")
        print("   largest exclusion category and they will fire your detector.\n")

    # ---- sanity check against the published rate ---------------------
    if times:
        span_min = (times[-1] - times[0]) / 60
        print(f"span {span_min:.0f} min, {len(times)} entries "
              f"({len(times) / max(span_min, 1) * 70:.0f} per 70-min match)")
        print("   Reference: McColgan et al. coded 2172 kickouts across 89 games, "
              "about 24 per game.\n")

    # ---- choose windows ---------------------------------------------
    win = pick_windows(times, args.window_minutes * 60, args.max_windows, args.pad_seconds)
    if not win:
        raise SystemExit("no windows found — check the timestamp file")

    rows, gt_rows = [], []
    total = 0
    for w, (start, end, idxs) in enumerate(win, 1):
        vid = f"{args.video_id}_w{w}"
        total += len(idxs)
        print(f"WINDOW {w}  ->  video_id: {vid}")
        print(f"  source {hhmmss(start)} to {hhmmss(end)}  "
              f"({(end - start) / 60:.1f} min, {len(idxs)} kickouts)")
        print(f"  config.yaml:  segment_start: \"{hhmmss(start)[:8]}\"  "
              f"segment_duration: \"{hhmmss(end - start)[:8]}\"")
        rows.append(dict(window=w, video_id=vid, source_start_s=start, source_end_s=end,
                         duration_s=end - start, n_kickouts=len(idxs),
                         segment_start=hhmmss(start)[:8],
                         segment_duration=hhmmss(end - start)[:8]))
        print("  kickouts, converted to CLIP-LOCAL seconds:")
        for n, j in enumerate(sorted(idxs), 1):
            local = times[j] - start
            print(f"    {n:2d}. source {labels[j]:>10}  ->  clip t = {local:7.2f} s")
            gt_rows.append(dict(video_id=vid, event_id=f"ko_{n:03d}",
                                source_timestamp=labels[j], source_s=times[j],
                                t_peak_s_approx=round(local, 2),
                                t_start_s="", t_end_s="", t_peak_s="",
                                n_players_in_contest="", outcome="", restart_type="",
                                coder_id="", coding_pass=1, visibility="", notes=""))
        print()

    uncovered = len(times) - total
    if uncovered:
        print(f"note: {uncovered} kickouts fall outside the chosen windows. That is fine — "
              f"say in the report which passages you analysed and why.\n")

    outdir = ROOT / "data" / "gt"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "segment_plan.csv", index=False)
    for vid, g in pd.DataFrame(gt_rows).groupby("video_id"):
        d = outdir / str(vid)
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["video_id"]).to_csv(d / "gt_events_skeleton.csv", index=False)
        print(f"wrote {d / 'gt_events_skeleton.csv'}")

    print("\n`t_peak_s_approx` is your noted timestamp, converted. It is a STARTING POINT,")
    print("not ground truth — step frame by frame to find the real t_start, t_peak and")
    print("t_end, then save as gt_events.csv. Coding straight from the approximation")
    print("would mean validating the model against your own rough notes.")


if __name__ == "__main__":
    main()
