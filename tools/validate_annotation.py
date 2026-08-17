#!/usr/bin/env python3
"""Check hand annotation before any metric is computed from it.

A metric computed on broken ground truth is worse than no metric, because it
looks like a result. Everything here is a structural check on the labels
themselves - nothing is compared against the detector's output except the box
SIZE distribution, which is the one comparison that is a finding rather than a
check (if the human found many more small boxes than the detector did, that gap
is the small-tercile recall story).

Checks, in order of how badly a failure invalidates downstream numbers:

  1. MOT16 parses            - 9 numeric columns, no NaN, positive w/h
  2. frame indices align     - every keyframe the sampler emitted is present,
                               and nothing else is
  3. track ids are usable    - id = -1 means CVAT exported SHAPES not TRACKS,
                               which silently destroys every tracking metric
  4. per-track continuity    - a track that vanishes and returns is either an
                               ID error or a real occlusion; both need eyes
  5. geometry sanity         - boxes inside the frame, plausible aspect ratios

Usage:
    python tools/validate_annotation.py --video-id lgf26_final_w1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lib.config import load_config, out_dir, resolve  # noqa: E402
from lib.logging_setup import get_logger  # noqa: E402
from lib.schema import read_table  # noqa: E402

log = get_logger("validate_annotation")

MOT_COLS = ["frame", "id", "bb_left", "bb_top", "bb_w", "bb_h", "conf", "cls", "vis"]


def load_mot(path: Path) -> pd.DataFrame:
    """Read MOT16 gt.txt WITHOUT assuming a frame-numbering convention.

    s09 subtracts 1 because MOT16 is nominally 1-based. This export is not: the
    sampler wrote absolute working-clip frame indices, so the frame column is
    already 0-based. Which convention is in force is decided in
    check_frame_alignment() by testing against the manifest, not assumed here.
    """
    df = pd.read_csv(path, header=None, names=MOT_COLS)
    df["area"] = df.bb_w * df.bb_h
    df["x1"], df["y1"] = df.bb_left, df.bb_top
    df["x2"], df["y2"] = df.bb_left + df.bb_w, df.bb_top + df.bb_h
    return df


def check_parse(df: pd.DataFrame, block: str) -> list[str]:
    errs = []
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        errs.append(f"{block}: NaN in columns {bad}")
    nonpos = int(((df.bb_w <= 0) | (df.bb_h <= 0)).sum())
    if nonpos:
        errs.append(f"{block}: {nonpos} boxes with non-positive width or height")
    log.info("  %s: %d rows parsed, %d columns, %d frames, %d distinct ids",
             block, len(df), df.shape[1] - 4, df.frame.nunique(), df.id.nunique())
    return errs


def check_frame_alignment(df: pd.DataFrame, expected: np.ndarray, block: str) -> list[str]:
    """Do the annotated frames match the keyframes the sampler emitted?"""
    errs = []
    got = np.array(sorted(df.frame.unique()))
    # decide the numbering convention by which one actually lines up
    hits_0 = len(np.intersect1d(got, expected))
    hits_1 = len(np.intersect1d(got - 1, expected))
    convention = "0-based absolute" if hits_0 >= hits_1 else "1-based (MOT default)"
    frames = got if hits_0 >= hits_1 else got - 1
    log.info("  %s: frame numbering looks %s (%d/%d align)",
             block, convention, max(hits_0, hits_1), len(expected))

    missing = np.setdiff1d(expected, frames)
    extra = np.setdiff1d(frames, expected)
    if len(missing):
        errs.append(f"{block}: {len(missing)} of {len(expected)} keyframes have NO "
                    f"annotation (first missing {missing[:5].tolist()})")
    if len(extra):
        errs.append(f"{block}: {len(extra)} annotated frames are not sampler keyframes "
                    f"({extra[:5].tolist()})")
    step = np.diff(np.sort(frames))
    if len(step):
        log.info("  %s: frame step modal %d, range %d-%d",
                 block, int(pd.Series(step).mode().iloc[0]), int(step.min()), int(step.max()))
    return errs, convention


def check_track_ids(df: pd.DataFrame, block: str) -> list[str]:
    errs = []
    n_shape = int((df.id == -1).sum())
    if n_shape:
        errs.append(f"{block}: {n_shape} of {len(df)} rows ({100*n_shape/len(df):.0f}%) "
                    f"have id=-1. CVAT exports -1 for SHAPES; only TRACKS carry an id. "
                    f"Every tracking metric (IDF1, MOTA, ID switches) is uncomputable "
                    f"for these rows, and they cannot be repaired after the fact")
    real = df[df.id != -1]
    if not len(real):
        return errs
    ids = np.array(sorted(real.id.unique()))
    gaps = np.setdiff1d(np.arange(ids.min(), ids.max() + 1), ids)
    if len(gaps):
        errs.append(f"{block}: track ids are not contiguous — {len(gaps)} missing in "
                    f"[{ids.min()},{ids.max()}] ({gaps[:10].tolist()})")
    log.info("  %s: %d real track ids in [%d,%d]", block, len(ids), ids.min(), ids.max())
    return errs


def check_continuity(df: pd.DataFrame, step: int, block: str) -> pd.DataFrame:
    """Tracks that disappear and come back. Either ID errors or real occlusions."""
    real = df[df.id != -1]
    rows = []
    for tid, g in real.groupby("id"):
        f = np.array(sorted(g.frame.unique()))
        d = np.diff(f)
        breaks = np.where(d > step)[0]
        if len(breaks):
            rows.append(dict(
                block_id=block, track_id=int(tid), n_frames=len(f),
                first_frame=int(f[0]), last_frame=int(f[-1]),
                n_gaps=len(breaks),
                longest_gap_frames=int(d[breaks].max()),
                longest_gap_s=round(float(d[breaks].max()) / 25, 2),
                gap_starts=";".join(str(int(f[b])) for b in breaks[:6]),
            ))
    return pd.DataFrame(rows)


def check_geometry(df: pd.DataFrame, w: int, h: int, block: str) -> list[str]:
    errs = []
    oob = int(((df.x1 < -1) | (df.y1 < -1) | (df.x2 > w + 1) | (df.y2 > h + 1)).sum())
    if oob:
        log.warning("  %s: %d boxes extend outside the %dx%d frame "
                    "(normal for players cut by the edge, check it is not all of them)",
                    block, oob, w, h)
    ar = df.bb_h / df.bb_w.replace(0, np.nan)
    odd = int(((ar < 0.8) | (ar > 6.0)).sum())
    if odd:
        log.warning("  %s: %d boxes with height/width outside 0.8-6.0 "
                    "(a standing player is ~2-3); %.1f%% of rows",
                    block, odd, 100 * odd / len(df))
    log.info("  %s: bbox height median %.0f px (p10 %.0f, p90 %.0f), "
             "aspect median %.2f", block, df.bb_h.median(),
             df.bb_h.quantile(.1), df.bb_h.quantile(.9), ar.median())
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    fps = int(cfg["video"]["fps"])
    step = int(cfg["annotation"]["blocks"]["box_every_n_frames"])
    W, H = int(cfg["video"]["width"]), int(cfg["video"]["height"])

    gt_root = resolve(cfg, "gt") / args.video_id
    man = pd.read_csv(gt_root / "annotation_blocks" / "manifest.csv")
    blocks = pd.read_csv(gt_root / "annotation_blocks" / "blocks.csv")

    all_errs: list[str] = []
    per_block, gaps_all, gt_all = {}, [], []

    log.info("=== 1-5. structural checks ===")
    for b in blocks.itertuples():
        path = gt_root / "gt_boxes" / b.block_id / "gt.txt"
        if not path.exists():
            all_errs.append(f"{b.block_id}: {path} missing")
            continue
        df = load_mot(path)
        expected = man[man.block_id == b.block_id].frame_idx.values

        all_errs += check_parse(df, b.block_id)
        errs, conv = check_frame_alignment(df, expected, b.block_id)
        all_errs += errs
        if conv.startswith("1-based"):
            df["frame"] = df.frame - 1
        all_errs += check_track_ids(df, b.block_id)
        all_errs += check_geometry(df, W, H, b.block_id)

        g = check_continuity(df, step, b.block_id)
        if len(g):
            gaps_all.append(g)
        df["block_id"] = b.block_id
        gt_all.append(df)
        per_block[b.block_id] = df

    if not gt_all:
        raise SystemExit("no annotation found")
    gt = pd.concat(gt_all, ignore_index=True)

    # ---- 6. box counts per frame ---------------------------------------
    log.info("=== 6. boxes per annotated frame ===")
    for bid, df in per_block.items():
        bpf = df.groupby("frame").size()
        exp = man[man.block_id == bid]
        log.info("  %s: %.2f boxes/frame (median %d, range %d-%d) over %d annotated frames; "
                 "detector averaged %.2f over the same block",
                 bid, bpf.mean(), int(bpf.median()), int(bpf.min()), int(bpf.max()), len(bpf),
                 exp.n_boxes.mean())

    # ---- 7. size distribution vs the detector --------------------------
    log.info("=== 7. annotated box area vs detector box area (same frames) ===")
    od = out_dir(cfg)
    det = read_table("detections", od / "detections.parquet")
    det["area"] = (det.x2 - det.x1) * (det.y2 - det.y1)
    det["bb_h"] = det.y2 - det.y1
    frames = set(gt.frame.tolist())
    dsub = det[det.frame_idx.isin(frames)]

    q1, q2 = np.percentile(gt.area, [33.333, 66.667])
    log.info("  GT area terciles: small < %.0f px2 < medium < %.0f px2 < large", q1, q2)
    for name, d in [("annotated", gt.area), ("detector", dsub.area)]:
        log.info("  %-9s n=%6d  p10 %6.0f  p25 %6.0f  median %6.0f  p75 %6.0f  p90 %7.0f px2",
                 name, len(d), *[d.quantile(p) for p in (.1, .25, .5, .75, .9)])
    for name, d in [("annotated", gt.bb_h), ("detector", dsub.bb_h)]:
        log.info("  %-9s bbox HEIGHT p10 %5.0f  median %5.0f  p90 %5.0f px",
                 name, d.quantile(.1), d.median(), d.quantile(.9))

    gt_small = int((gt.area < q1).sum())
    det_small = int((dsub.area < q1).sum())
    log.info("  below the GT small threshold (%.0f px2): %d annotated vs %d detected "
             "-> the human found %.2fx as many small boxes",
             q1, gt_small, det_small, gt_small / max(det_small, 1))
    log.info("  overall: %d annotated boxes vs %d detections on the same %d frames "
             "(ratio %.2f)", len(gt), len(dsub), len(frames), len(gt) / max(len(dsub), 1))

    # ---- 8. continuity report ------------------------------------------
    log.info("=== 8. tracks that disappear and reappear ===")
    if gaps_all:
        gaps = pd.concat(gaps_all, ignore_index=True).sort_values(
            "longest_gap_frames", ascending=False)
        out = gt_root / "annotation_blocks" / "track_gaps.csv"
        gaps.to_csv(out, index=False)
        log.info("  %d tracks with at least one gap, longest %.2fs — written to %s",
                 len(gaps), gaps.longest_gap_s.max(), out)
        for r in gaps.head(10).itertuples():
            log.info("    %s id=%-4d %d gaps, longest %.2fs, from frame %s",
                     r.block_id, r.track_id, r.n_gaps, r.longest_gap_s, r.gap_starts)
        log.info("  A gap is an ID error OR a real occlusion. They are not separable "
                 "from the file alone — these need eyes on the clip.")
    else:
        log.info("  none")

    # ---- verdict ---------------------------------------------------------
    log.info("=== VERDICT ===")
    if all_errs:
        for e in all_errs:
            log.error("  FAIL %s", e)
        log.error("  %d problem(s). Metrics computed on this annotation are not "
                  "trustworthy until these are resolved or explicitly scoped around.",
                  len(all_errs))
        sys.exit(1)
    log.info("  all structural checks passed")


if __name__ == "__main__":
    main()
