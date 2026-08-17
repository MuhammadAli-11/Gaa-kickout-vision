#!/usr/bin/env python3
"""Choose which frames to hand-annotate, and say why for each one.

Two modes:

  --mode blocks  (default)
      TWO contiguous 60 s blocks, boxed every 5th frame, interpolated between
      in CVAT. Chosen because sparse frames cannot be interpolated and carry no
      track identity, so they can evaluate DETECTION but not TRACKING - and
      the tracking metrics (IDF1, MOTA, ID switches) are exactly the ones
      docs/09 is missing. Contiguous blocks buy those at the cost of scene
      diversity, which is the right trade at this budget.

  --mode sparse
      The original 300-frame area-stratified sample. Retained because it gives
      wider scene coverage per box drawn, which is the better buy if only
      detection is being measured.

The sampling decisions here ARE the experiment design, so each is recorded in
the manifest rather than left to be reconstructed later:

  Stratified by detected box area, oversampling small.
      Recall collapses in the small tercile, so a uniform sample spends most of
      the annotation budget on boxes the detector already gets right.

  A quota of frames from BETWEEN kickouts.
      A sample drawn only from contest windows cannot measure a false positive
      in ordinary play, because ordinary play is not in it.

  Slow-motion and replay intervals excluded.
      s05 assumes one frame is 1/25 s of real time. Frames where that is false
      should not become ground truth for a detector evaluated on the assumption.

  train/val split by TIME BLOCK, never by frame.
      Adjacent frames are near-duplicates. A random frame split puts a frame's
      own neighbours in the other half, which inflates val scores without any
      generalisation behind it. Blocks are contiguous seconds, split whole.

Sampling uses DETECTED boxes to stratify, which is circular in one direction:
a player the detector missed cannot pull its frame into the small stratum. The
oversample factor is the hedge, and the limitation belongs in the write-up.

Usage:
    python tools/sample_frames_for_annotation.py --video-id lgf26_final_w1 \\
        --exclude data/gt/lgf26_final_w1/excluded_intervals.csv
    python tools/sample_frames_for_annotation.py --video-id lgf26_final_w1 \\
        --mode sparse --exclude ... --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lib.config import load_config, out_dir, resolve  # noqa: E402
from lib.logging_setup import get_logger  # noqa: E402
from lib.schema import read_table, write_table  # noqa: E402

log = get_logger("sample_frames")


def load_exclusions(path: Path | None, fps: int) -> list[tuple[float, float]]:
    """Intervals (start_s,end_s) to keep out of the sample entirely."""
    if not path or not path.exists():
        return []
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    s = cols.get("start_s") or cols.get("start")
    e = cols.get("end_s") or cols.get("end")
    if not s or not e:
        raise SystemExit(f"{path} needs start_s and end_s columns")
    return [(float(r[s]), float(r[e])) for _, r in df.iterrows()]


def guard_holdout(cfg: dict, video_id: str) -> None:
    """Refuse to sample the held-out test video, whatever the caller asked for.

    This is a hard stop rather than a warning on purpose: the value of an
    untouched test set is destroyed by a single careless run, and it cannot be
    restored afterwards. Cheap to enforce here, impossible to undo later.
    """
    forbidden = [str(v) for v in cfg["annotation"].get("holdout_video_ids", [])]
    if video_id in forbidden:
        raise SystemExit(
            f"REFUSED: '{video_id}' is listed in annotation.holdout_video_ids. "
            "It is the held-out test set and must not be inspected, sampled or "
            "annotated. Remove it from that list deliberately if this is really "
            "what you intend.")


def choose_blocks(per: pd.DataFrame, feat: pd.DataFrame, peaks: list[float],
                  excl: list[tuple[float, float]], cfg: dict) -> pd.DataFrame:
    """Pick n contiguous blocks, each containing a kickout, none overlapping a
    replay/slow-motion candidate, preferring those that also contain a cut.

    Returns one row per chosen block with the reason it was chosen, so the
    manifest records the design rather than only its output.
    """
    a = cfg["annotation"]
    b = a["blocks"]
    span = float(a["block_s"])
    stride = float(b["search_stride_s"])
    dur = float(feat.timestamp_s.max())

    rows = []
    start = 0.0
    while start + span <= dur:
        end = start + span
        ko = [p for p in peaks if start <= p < end]
        overlap = sum(max(0.0, min(end, e) - max(start, s)) for s, e in excl)
        win = feat[(feat.timestamp_s >= start) & (feat.timestamp_s < end)]
        sub = per[(per.t_s >= start) & (per.t_s < end)]
        rows.append(dict(
            t_start_s=start, t_end_s=end, n_kickouts=len(ko),
            kickout_times=";".join(f"{p:.1f}" for p in ko),
            excl_overlap_s=round(overlap, 2),
            churn_flags=int(win.scene_change_flag.sum()),
            n_frames_with_boxes=len(sub),
            mean_boxes_per_frame=round(sub.n_boxes.mean(), 2) if len(sub) else 0.0,
            mean_small_frac=round(sub.small_frac.mean(), 3) if len(sub) else 0.0,
        ))
        start += stride
    cand = pd.DataFrame(rows)
    log.info("%d candidate blocks of %.0fs at %.0fs stride", len(cand), span, stride)

    if b["require_kickout_per_block"]:
        cand = cand[cand.n_kickouts >= 1]
        log.info("  %d contain at least one kickout", len(cand))
    if b["forbid_exclusion_overlap"]:
        cand = cand[cand.excl_overlap_s == 0]
        log.info("  %d of those overlap no replay/slow-motion candidate", len(cand))
    if cand.empty:
        raise SystemExit("no block satisfies the constraints — relax "
                         "annotation.blocks or widen the search stride")

    # Rank: more kickouts first, then more small boxes (the hard cases), then
    # more churn. Ranking is only a tie-break; the constraints do the work.
    cand = cand.sort_values(["n_kickouts", "mean_small_frac", "churn_flags"],
                            ascending=False)

    chosen: list[dict] = []
    for _, r in cand.iterrows():
        if len(chosen) >= int(b["n_blocks"]):
            break
        if any(r.t_start_s < c["t_end_s"] and c["t_start_s"] < r.t_end_s for c in chosen):
            continue                      # no overlapping blocks
        chosen.append(r.to_dict())

    if len(chosen) < int(b["n_blocks"]):
        log.warning("wanted %d blocks, only %d satisfy the constraints",
                    b["n_blocks"], len(chosen))

    out = pd.DataFrame(chosen).sort_values("t_start_s").reset_index(drop=True)
    out["block_id"] = [f"blk_{i:02d}" for i in range(1, len(out) + 1)]
    out["reason"] = [
        f"{int(r.n_kickouts)} kickout(s) at {r.kickout_times}s; no overlap with a "
        f"replay/slow-motion candidate; {int(r.churn_flags)} track-churn flags; "
        f"{r.mean_boxes_per_frame:.1f} boxes/frame, {100*r.mean_small_frac:.0f}% small-tercile"
        for r in out.itertuples()]

    n_cut = int((out.churn_flags > 0).sum())
    if n_cut < int(b["min_blocks_with_cut"]):
        log.warning("min_blocks_with_cut=%d but only %d chosen block(s) show any "
                    "track churn", b["min_blocks_with_cut"], n_cut)
    else:
        log.warning("cut requirement judged on scene_change_flag, an UNVALIDATED "
                    "track-churn proxy. A pixel-level scan of the chosen blocks found "
                    "no hard cut separable from fast panning (top inter-frame "
                    "differences reach only 1.1-1.3x p99). Treat 'contains a camera "
                    "cut' as UNCONFIRMED until an annotator says otherwise.")
    return out


def emit_blocks(sel_blocks: pd.DataFrame, per: pd.DataFrame, cfg: dict,
                video_id: str, rng: np.random.Generator, dry_run: bool,
                force: bool = False) -> None:
    """Extract keyframes per block, one CVAT task per block, and write manifests."""
    a, fps = cfg["annotation"], int(cfg["video"]["fps"])
    b = a["blocks"]
    step = int(b["box_every_n_frames"])

    # val split by BLOCK. With two blocks the only honest split is 1/1, which
    # makes val a single 60 s passage - weak, and said so rather than hidden.
    n_val = min(int(b["n_val_blocks"]), max(len(sel_blocks) - 1, 0))
    val_ids = set(rng.choice(sel_blocks.block_id.values, size=n_val, replace=False).tolist()) \
        if n_val else set()
    sel_blocks["split"] = np.where(sel_blocks.block_id.isin(val_ids), "val", "train")

    frames = []
    for r in sel_blocks.itertuples():
        f0, f1 = int(round(r.t_start_s * fps)), int(round(r.t_end_s * fps))
        for fi in range(f0, f1, step):
            row = per[per.frame_idx == fi]
            frames.append(dict(
                block_id=r.block_id, split=r.split, frame_idx=fi, t_s=fi / fps,
                n_boxes=int(row.n_boxes.iloc[0]) if len(row) else 0,
                small_frac=float(row.small_frac.iloc[0]) if len(row) else np.nan,
                is_keyframe=1))
    fr = pd.DataFrame(frames)

    est_boxes = int(fr.n_boxes.sum())
    log.info("%d blocks -> %d keyframes (every %dth frame), ~%d boxes to draw",
             len(sel_blocks), len(fr), step, est_boxes)
    for r in sel_blocks.itertuples():
        n = int((fr.block_id == r.block_id).sum())
        log.info("  %s [%.0f,%.0f)s  %s  %d keyframes  ~%d boxes  | %s",
                 r.block_id, r.t_start_s, r.t_end_s, r.split.upper(), n,
                 int(fr[fr.block_id == r.block_id].n_boxes.sum()), r.reason)

    out_root = resolve(cfg, "gt") / video_id / "annotation_blocks"
    if dry_run:
        log.info("dry run — no images extracted, no manifest written")
        return

    # The manifest is a place humans write things back: which candidate cut was
    # real, which frames were unannotatable, notes from the annotation session.
    # Rewriting it unconditionally destroys that with no warning and no way to
    # recover it. Refuse instead, unless the caller says to overwrite.
    existing = out_root / "manifest.csv"
    if existing.exists() and not force:
        prior = pd.read_csv(existing)
        extra = [c for c in prior.columns if c not in fr.columns]
        raise SystemExit(
            f"REFUSED: {existing} already exists"
            + (f" and carries hand-added column(s) {extra}" if extra else "")
            + ".\nRe-running would overwrite whatever was recorded in it. Pass "
              "--force if the plan really should be regenerated from scratch, "
              "after copying anything you want to keep.")

    clip = resolve(cfg, "clips") / f"{video_id}_working.mp4"
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")

    # One sequential decode per block rather than one seek per keyframe. A block
    # is contiguous, so seeking 300 times to read every 5th frame re-decodes the
    # same GOPs 300 times over; reading straight through is both faster and not
    # subject to seek landing on the wrong frame near a keyframe boundary.
    import cv2
    for r in sel_blocks.itertuples():
        d = out_root / r.block_id / "images"
        d.mkdir(parents=True, exist_ok=True)
        sub = fr[fr.block_id == r.block_id]
        wanted = set(sub.frame_idx.tolist())
        f0, f1 = min(wanted), max(wanted)

        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {clip}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        written = 0
        for fi in range(f0, f1 + 1):
            ok, frame = cap.read()
            if not ok:
                log.warning("  %s: decode ended early at frame %d", r.block_id, fi)
                break
            if fi not in wanted:
                continue
            dest = d / f"{fi:06d}.jpg"
            if not dest.exists():
                cv2.imwrite(str(dest), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(a["jpeg_quality_pct"])])
            written += 1
        cap.release()
        if written != len(sub):
            log.warning("  %s: wrote %d of %d expected keyframes", r.block_id, written, len(sub))
        log.info("  %s: %d images -> %s", r.block_id, written, d)

    out_root.mkdir(parents=True, exist_ok=True)
    write_table(sel_blocks, "annotation_blocks", out_root / "blocks.csv")
    write_table(fr, "annotation_keyframes", out_root / "manifest.csv")

    import json
    (out_root / "annotation_plan.json").write_text(json.dumps({
        "video_id": video_id,
        "mode": "contiguous_blocks",
        "block_length_s": float(a["block_s"]),
        "box_every_n_frames": step,
        "fps": fps,
        "n_blocks": len(sel_blocks),
        "keyframes_total": len(fr),
        "estimated_boxes": est_boxes,
        "val_blocks": sorted(val_ids),
        "split_unit": "time_block (never by frame: adjacent frames are "
                      "near-duplicates and a frame split leaks val into train)",
        "exclusions_applied": "data/gt/%s/excluded_intervals.csv — blocks "
                              "overlapping any candidate replay/slow-motion "
                              "interval were rejected outright" % video_id,
        "cut_requirement_status": "UNCONFIRMED — judged on scene_change_flag "
                                  "(unvalidated track-churn proxy). Pixel scan "
                                  "found no hard cut cleanly separable from "
                                  "fast panning.",
        "holdout_never_sampled": cfg["annotation"].get("holdout_video_ids", []),
        "frames_covered_by_interpolation": len(fr) * step,
        "estimated_hours": {f"{r} s/box": round(est_boxes * r / 3600, 1)
                            for r in (1.5, 2.5, 4.0)},
        "estimated_hours_note":
            "est_boxes counts DETECTED boxes, so it under-counts wherever the "
            "detector missed a player - which is the small tercile, which is the "
            "point of annotating. Treat these hours as a floor, not a budget.",
        "blocks": sel_blocks.to_dict(orient="records"),
    }, indent=2, default=str))

    log.info("effort: %d boxes -> %.1f h at 1.5 s/box, %.1f h at 2.5, %.1f h at 4.0",
             est_boxes, est_boxes * 1.5 / 3600, est_boxes * 2.5 / 3600, est_boxes * 4.0 / 3600)
    log.info("  %d keyframes propagate to %d frames (%.0fs continuously labelled)",
             len(fr), len(fr) * step, len(fr) * step / fps)

    log.info("wrote %s", out_root / "annotation_plan.json")
    log.info("CVAT: create ONE TASK PER BLOCK (never one task spanning both — "
             "interpolation must not cross a block boundary).")
    log.info("  upload %s/<block_id>/images, single label 'player', "
             "annotate as TRACKS not shapes so interpolation applies,", out_root)
    log.info("  then export MOT 1.1 to %s/<block_id>/gt_boxes.txt",
             resolve(cfg, "gt") / video_id)
    log.info("VAL BLOCK(S): %s — keep whole when fine-tuning.", sorted(val_ids) or "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--mode", default="blocks", choices=["blocks", "sparse"])
    ap.add_argument("--exclude", default=None,
                    help="CSV of slow-motion/replay intervals to skip")
    ap.add_argument("--n", type=int, default=None, help="override annotation.n_frames")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="plan only, extract no images")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest, discarding anything hand-added to it")
    args = ap.parse_args()

    cfg = load_config(args.config)
    guard_holdout(cfg, args.video_id)
    a, fps = cfg["annotation"], int(cfg["video"]["fps"])
    n_target = int(args.n or a["n_frames"])
    rng = np.random.default_rng(args.seed)
    od = out_dir(cfg)

    det = read_table("detections", od / "detections.parquet")
    det["area"] = (det.x2 - det.x1) * (det.y2 - det.y1)
    q1, q2 = np.percentile(det.area, [33.333, 66.667])
    log.info("area terciles from %d detections: small < %.0f < medium < %.0f < large",
             len(det), q1, q2)

    det["is_small"] = det.area < q1
    per = det.groupby("frame_idx").agg(
        n_boxes=("area", "size"), n_small=("is_small", "sum"),
        median_area=("area", "median"), mean_conf=("conf", "mean"),
    ).reset_index()
    per["t_s"] = per.frame_idx / fps
    per["small_frac"] = per.n_small / per.n_boxes
    # Blocks mode keeps every frame: a block is contiguous by definition, so a
    # sparse frame's "too few boxes to be worth drawing" test does not apply.
    per_all = per.copy()
    per = per[per.n_boxes >= a["min_boxes_per_frame"]]
    log.info("%d frames with >= %d boxes", len(per), a["min_boxes_per_frame"])

    # ---- contest windows from the kickout skeleton -----------------------
    sk = resolve(cfg, "gt") / args.video_id / "gt_events_skeleton.csv"
    peaks = []
    if sk.exists():
        peaks = pd.read_csv(sk).t_peak_s_approx.dropna().tolist()
    pad = float(a["contest_pad_s"])
    in_contest = np.zeros(len(per), dtype=bool)
    for p in peaks:
        in_contest |= (per.t_s >= p - pad) & (per.t_s <= p + pad)
    per["in_contest"] = in_contest
    log.info("%d kickout peaks -> %d frames inside a +/-%.0fs contest window "
             "(%.1f%% of candidates)", len(peaks), int(in_contest.sum()), pad,
             100 * in_contest.mean())

    # ---- exclusions ------------------------------------------------------
    excl = load_exclusions(Path(args.exclude) if args.exclude else None, fps)
    if excl:
        drop = np.zeros(len(per), dtype=bool)
        for s, e in excl:
            drop |= (per.t_s >= s) & (per.t_s <= e)
        log.info("excluding %d frames inside %d slow-motion/replay intervals",
                 int(drop.sum()), len(excl))
        per = per[~drop]
    else:
        log.warning("no --exclude file given: slow-motion and replay frames are "
                    "NOT being filtered. Pass one before annotating for real.")

    if args.mode == "blocks":
        feat = read_table("features", od / "features.parquet")
        sel_blocks = choose_blocks(per_all, feat, peaks, excl, cfg)
        emit_blocks(sel_blocks, per_all, cfg, args.video_id, rng, args.dry_run, args.force)
        return

    # ---- time blocks, split whole ---------------------------------------
    per["block"] = (per.t_s // a["block_s"]).astype(int)
    blocks = np.array(sorted(per.block.unique()))
    n_val = max(1, int(round(len(blocks) * a["val_block_frac"])))
    val_blocks = set(rng.choice(blocks, size=n_val, replace=False).tolist())
    per["split"] = np.where(per.block.isin(val_blocks), "val", "train")
    log.info("%d blocks of %.0fs -> %d val blocks %s",
             len(blocks), a["block_s"], n_val, sorted(val_blocks))

    # ---- weights: oversample small-rich, guarantee outside-contest -------
    w = 1.0 + (a["small_oversample"] - 1.0) * per.small_frac.values
    per["weight"] = w / w.sum()

    n_out_min = int(round(n_target * a["min_outside_contest"]))
    picks = []
    for label, sub, quota in [
        ("outside_contest", per[~per.in_contest], n_out_min),
        ("contest", per[per.in_contest], n_target - n_out_min),
    ]:
        if not len(sub):
            log.warning("no candidate frames for stratum %s", label)
            continue
        k = min(quota, len(sub))
        p = sub.weight.values / sub.weight.values.sum()
        idx = rng.choice(len(sub), size=k, replace=False, p=p)
        got = sub.iloc[idx].copy()
        got["stratum"] = label
        picks.append(got)
        if k < quota:
            log.warning("stratum %s: wanted %d, only %d available", label, quota, k)

    sel = pd.concat(picks).sort_values("frame_idx").reset_index(drop=True)
    sel["reason"] = np.where(
        sel.small_frac >= sel.small_frac.median(),
        "small-tercile-rich; " + sel.stratum,
        "size-diverse; " + sel.stratum)

    # ---- report ----------------------------------------------------------
    log.info("selected %d frames", len(sel))
    log.info("  outside contest: %d (%.0f%%)  | inside: %d",
             int((sel.stratum == "outside_contest").sum()),
             100 * (sel.stratum == "outside_contest").mean(),
             int((sel.stratum == "contest").sum()))
    log.info("  split: train %d / val %d frames",
             int((sel.split == "train").sum()), int((sel.split == "val").sum()))
    log.info("  mean small_frac in sample %.3f vs %.3f in the candidate pool "
             "(oversample working if higher)", sel.small_frac.mean(), per.small_frac.mean())
    log.info("  est. boxes to draw: %d", int(sel.n_boxes.sum()))

    out_root = resolve(cfg, "gt") / args.video_id / "annotation"
    man_path = out_root / "manifest.csv"
    if args.dry_run:
        log.info("dry run — no images extracted, no manifest written")
        return

    (out_root / "images").mkdir(parents=True, exist_ok=True)
    clip = resolve(cfg, "clips") / f"{args.video_id}_working.mp4"
    for _, r in sel.iterrows():
        dest = out_root / "images" / f"{int(r.frame_idx):06d}.jpg"
        if dest.exists():
            continue
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y",
             "-ss", f"{r.frame_idx / fps:.3f}", "-i", str(clip),
             "-frames:v", "1", "-q:v", str(a["jpeg_quality"]), str(dest)],
            check=True)

    sel_out = sel[["frame_idx", "t_s", "block", "split", "stratum", "reason",
                   "n_boxes", "n_small", "small_frac", "median_area", "mean_conf"]]
    sel_out.to_csv(man_path, index=False)
    log.info("wrote %s and %d images", man_path, len(sel))
    log.info("CVAT: create a task, upload %s, single label 'player', then export "
             "MOT 1.1 to %s", out_root / "images",
             resolve(cfg, "gt") / args.video_id / "gt_boxes.txt")
    log.info("The val BLOCKS are %s — keep them whole when you fine-tune.",
             sorted(val_blocks))


if __name__ == "__main__":
    main()
