#!/usr/bin/env python3
"""S09 — Failure audit. Export a clip around every error so it can be classified.

The taxonomy is fixed in advance (docs/09) so that classification is
coding, not storytelling:
    aerial_occlusion | camera_pan_cut | scale_distance | no_ball_ambiguity
    | annotation_error | other

Produces one clip per FP/FN plus a pre-filled CSV with an empty `cause`
column for you to complete. The completed CSV is a RESULT, and the
cross-tab of cause by error type is the most quotable table in the report.

Usage:
    python src/s09_failure_audit.py --video-id gaa_lf_r3
    # then fill in outputs/<run_id>/failure_audit.csv by hand
"""
from __future__ import annotations

import argparse
import subprocess

import pandas as pd

from lib.config import load_config, resolve, out_dir
from lib.logging_setup import get_logger
from lib.schema import write_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--no-clips", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s09_audit", od / "logs")
    pad = cfg["audit"]["clip_pad_s"]
    clip = resolve(cfg, "clips") / f"{args.video_id}_working.mp4"
    fail_dir = od / "failures"; fail_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(od / "event_matches.csv")
    errs = m[m.kind.isin(["FP", "FN"])].copy()
    # Near-misses matter too: a prediction that overlaps but misses the
    # tIoU bar is a boundary problem, not a detection problem, and lumping
    # the two together hides the real error budget.
    near = m[(m.kind == "FP") & (m.tiou > 0.1)]
    log.info("%d FP, %d FN (%d of the FPs overlap a real event -> boundary errors)",
             int((m.kind == "FP").sum()), int((m.kind == "FN").sum()), len(near))

    rows = []
    for i, r in enumerate(errs.itertuples(), 1):
        t = r.t_peak_pred if r.kind == "FP" else r.t_peak_gt
        kind = "boundary" if (r.kind == "FP" and r.tiou > 0.1) else r.kind
        name = f"{kind}_{i:02d}_t{t:07.2f}.mp4".replace(".", "-", 1)
        path = fail_dir / name
        if not args.no_clips:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-ss", str(max(t - pad, 0)), "-i", str(clip),
                            "-t", str(2 * pad), "-c:v", "libx264", "-crf", "23", str(path)],
                           check=False)
        rows.append(dict(case_id=f"case_{i:03d}", error_type=kind, t_center_s=float(t),
                         clip_path=str(path.relative_to(od)), cause="", secondary_cause="",
                         reviewer_note=""))

    df = pd.DataFrame(rows)
    write_table(df, "failure_audit", od / "failure_audit.csv")
    log.info("exported %d failure clips to %s", len(df), fail_dir)
    log.info("NEXT: watch each clip, fill the `cause` column from the fixed taxonomy "
             "(docs/09), then re-run s11 to build the cause cross-tab.")


if __name__ == "__main__":
    main()
