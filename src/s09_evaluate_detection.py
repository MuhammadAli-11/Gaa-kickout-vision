#!/usr/bin/env python3
"""S06 — Detection and tracking metrics on the annotated frame subset.

Detection : P, R, F1 @ IoU 0.5; mAP@0.5; mAP@0.5:0.95; recall stratified
            by bbox-area tercile (small/medium/large).
Tracking  : MOTA, IDF1, ID switches, fragmentations, MT/ML via motmetrics.

The size stratification is the headline. Broadcast GAA is played on a
pitch up to 145 m long; far-side players fall below the size at which a
COCO-pretrained detector is reliable, and the tercile table is the
cleanest evidence for that.

Usage:
    python src/s06_evaluate_detection.py --video-id gaa_lf_r3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir
from lib.logging_setup import get_logger
from lib.matching import box_iou
from lib.schema import read_table
from lib.statsx import prf, average_precision, wilson_ci


MOT_COLS = ["frame", "id", "bb_left", "bb_top", "bb_w", "bb_h", "conf", "cls", "vis"]


def load_mot_gt(path, expected_frames=None) -> pd.DataFrame:
    """Read a MOT16 gt.txt, DETECTING the frame-numbering convention.

    MOT16 is nominally 1-based, and this function used to subtract 1
    unconditionally. That is wrong for an export whose frame column holds
    absolute working-clip indices, as the block sampler produces: subtracting 1
    shifts every box to the neighbouring frame, which does not fail loudly - it
    quietly costs a little IoU on every single box and looks like a mediocre
    detector. When `expected_frames` is supplied the convention is decided by
    which reading actually lines up, rather than assumed.
    """
    gt = pd.read_csv(path, header=None, names=MOT_COLS)
    gt = gt[gt.conf > 0] if gt.conf.notna().any() else gt
    raw = gt.frame.astype(int)
    if expected_frames is not None:
        exp = np.asarray(expected_frames)
        one_based = len(np.intersect1d(raw - 1, exp)) > len(np.intersect1d(raw, exp))
    else:
        one_based = True
    gt["frame_idx"] = raw - 1 if one_based else raw
    gt.attrs["frame_convention"] = "1-based" if one_based else "0-based absolute"
    gt["x1"], gt["y1"] = gt.bb_left, gt.bb_top
    gt["x2"], gt["y2"] = gt.bb_left + gt.bb_w, gt.bb_top + gt.bb_h
    gt["area"] = gt.bb_w * gt.bb_h
    return gt


def load_gt_blocks(cfg: dict, video_id: str, log) -> pd.DataFrame:
    """Per-block gt_boxes/<block_id>/gt.txt, or the legacy single gt_boxes.txt.

    Track ids restart at 1 in every block, so `id` alone is not unique across
    blocks. A globally unique `gid` is built here; the tracking metrics are
    still accumulated per block, because the blocks are 120 s apart and an
    identity cannot meaningfully persist across that gap.
    """
    root = resolve(cfg, "gt") / video_id
    blk_dir = root / "gt_boxes"
    man_path = root / "annotation_blocks" / "manifest.csv"
    man = pd.read_csv(man_path) if man_path.exists() else None

    if blk_dir.is_dir() and any(blk_dir.glob("*/gt.txt")):
        frames = []
        for d in sorted(p for p in blk_dir.iterdir() if p.is_dir()):
            exp = man[man.block_id == d.name].frame_idx.values if man is not None else None
            g = load_mot_gt(d / "gt.txt", exp)
            g["block_id"] = d.name
            log.info("  %s: %d boxes, %d frames, frame numbering %s",
                     d.name, len(g), g.frame_idx.nunique(), g.attrs["frame_convention"])
            frames.append(g)
        gt = pd.concat(frames, ignore_index=True)
    else:
        gt = load_mot_gt(root / "gt_boxes.txt")
        gt["block_id"] = "all"

    gt["gid"] = gt.block_id + "_" + gt.id.astype(int).astype(str)
    n_shape = int((gt.id == -1).sum())
    if n_shape:
        log.warning("%d of %d GT rows (%.0f%%) have id=-1 (CVAT SHAPES, not tracks). "
                    "They are kept for DETECTION metrics and excluded from TRACKING "
                    "metrics, which need an identity to score.",
                    n_shape, len(gt), 100 * n_shape / len(gt))
    return gt


def match_frame(pred_boxes, gt_boxes, thr):
    """Greedy highest-IoU matching within one frame."""
    M = box_iou(pred_boxes, gt_boxes)
    used_g, pairs = set(), []
    for i in np.argsort(-M.max(axis=1)) if M.size else []:
        j = int(np.argmax(np.where(np.isin(np.arange(M.shape[1]), list(used_g)), -1, M[i])))
        if M[i, j] >= thr and j not in used_g:
            used_g.add(j); pairs.append((int(i), j, float(M[i, j])))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s06_eval_det", od / "logs")
    thr = cfg["eval"]["det_iou_thresh"]
    fps = int(cfg["video"]["fps"])

    gt = load_gt_blocks(cfg, args.video_id, log)
    det = read_table("detections", od / "detections.parquet")
    frames = sorted(gt.frame_idx.unique())
    det = det[det.frame_idx.isin(frames)]
    log.info("annotated subset: %d frames, %d GT boxes, %d predictions", len(frames), len(gt), len(det))

    # ---- annotation tightness, BEFORE any metric -------------------------
    # IoU is a ratio of areas, so a systematically loose box caps every score
    # regardless of detector quality. Aspect ratio is the cheapest tell: a
    # standing person's box is roughly 2-3x taller than wide.
    gt_ar = (gt.bb_h / gt.bb_w.replace(0, np.nan)).median()
    det_ar = ((det.y2 - det.y1) / (det.x2 - det.x1).replace(0, np.nan)).median()
    log.info("box aspect (h/w): GT median %.2f vs detector median %.2f", gt_ar, det_ar)
    if gt_ar < 0.75 * det_ar:
        log.warning("GT boxes are proportionally %.0f%% WIDER than the detector's for the "
                    "same class of object. This is a box-tightness problem in the "
                    "ANNOTATION and it caps IoU independently of detector quality. "
                    "Treat every number below as a lower bound, and see the IoU "
                    "sensitivity sweep before quoting anything.",
                    100 * (det_ar / gt_ar - 1))

    # ---- terciles from GT box area (fixed once, reported in the paper) ----
    q1, q2 = np.percentile(gt.area, [33.3, 66.7])
    gt["size_bin"] = pd.cut(gt.area, [-np.inf, q1, q2, np.inf], labels=["small", "medium", "large"])
    log.info("size terciles by GT bbox area: small<%.0f px2, medium<%.0f px2", q1, q2)
    for s in ["small", "medium", "large"]:
        sub = gt[gt.size_bin == s]
        if len(sub):
            log.info("  %-6s n=%5d  bbox height median %5.1f px (p10 %.0f, p90 %.0f)",
                     s, len(sub), sub.bb_h.median(), sub.bb_h.quantile(.1), sub.bb_h.quantile(.9))

    scores, labels = [], []
    tp = fp = 0
    hit_by_size = {"small": 0, "medium": 0, "large": 0}
    n_by_size = gt.size_bin.value_counts().to_dict()

    for f in frames:
        p = det[det.frame_idx == f].sort_values("conf", ascending=False)
        g = gt[gt.frame_idx == f]
        pairs = match_frame(p[["x1", "y1", "x2", "y2"]].values, g[["x1", "y1", "x2", "y2"]].values, thr)
        matched_p = {i for i, _, _ in pairs}
        for i, j, _ in pairs:
            hit_by_size[str(g.iloc[j].size_bin)] += 1
        tp += len(pairs); fp += len(p) - len(pairs)
        for i in range(len(p)):
            scores.append(float(p.iloc[i].conf)); labels.append(1 if i in matched_p else 0)

    fn = len(gt) - tp
    res = prf(tp, fp, fn, cfg["eval"]["alpha"])
    ap50, rec, prec = average_precision(scores, labels, n_gt=len(gt))

    metrics = {
        "n_frames_annotated": len(frames), "n_gt_boxes": int(len(gt)),
        "tp": tp, "fp": fp, "fn": fn,
        **{k: v.to_dict() for k, v in res.items()},
        "mAP@0.5": ap50,
        "recall_by_size": {
            s: {"recall": hit_by_size[s] / n_by_size.get(s, 1),
                "n": int(n_by_size.get(s, 0)),
                "ci": list(wilson_ci(hit_by_size[s], int(n_by_size.get(s, 0))))
                if n_by_size.get(s) else [np.nan, np.nan]}
            for s in ["small", "medium", "large"]},
        "size_tercile_edges_px2": [float(q1), float(q2)],
    }

    # ---- IoU sensitivity: how much of the score is annotation tightness? --
    # If the deficit were detector misses, lowering the IoU bar would not help
    # much. If it is loose boxes, it helps a lot. The sweep separates them and
    # costs one extra pass.
    sweep = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        tp_t = 0
        for f in frames:
            p = det[det.frame_idx == f]
            g = gt[gt.frame_idx == f]
            tp_t += len(match_frame(p[["x1", "y1", "x2", "y2"]].values,
                                    g[["x1", "y1", "x2", "y2"]].values, t))
        sweep[f"{t:.1f}"] = {"tp": tp_t, "recall": tp_t / len(gt),
                             "precision": tp_t / max(len(det), 1)}
    metrics["iou_sweep"] = sweep
    log.info("IoU sweep (recall): " + "  ".join(
        f"{k}:{v['recall']:.3f}" for k, v in sweep.items()))

    # ---- tracking metrics ------------------------------------------
    # Accumulated PER BLOCK. Track ids restart at 1 in each block and the blocks
    # are 120 s apart, so a single accumulator would score every id as having
    # teleported between blocks and invent switches that never happened.
    try:
        # motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.0. The
        # shim restores the exact pre-2.0 behaviour (asarray with a float dtype)
        # rather than pinning NumPy back, which would drag the whole pipeline
        # to an older stack for one deleted alias.
        if not hasattr(np, "asfarray"):
            np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
        import motmetrics as mm
        tr = read_table("tracks", od / "tracks.parquet")
        gt_tr = gt[gt.id != -1]
        if not len(gt_tr):
            raise ValueError("no GT rows carry a track id (all are CVAT shapes)")

        mh = mm.metrics.create()
        want = ["mota", "idf1", "num_switches", "num_fragmentations",
                "mostly_tracked", "mostly_lost", "num_unique_objects"]
        accs, names, switch_frames = [], [], []
        for bid, gb in gt_tr.groupby("block_id"):
            acc = mm.MOTAccumulator(auto_id=False)
            bframes = sorted(gb.frame_idx.unique())
            for f in bframes:
                g = gb[gb.frame_idx == f]
                t = tr[tr.frame_idx == f]
                dist = mm.distances.iou_matrix(
                    g[["bb_left", "bb_top", "bb_w", "bb_h"]].values,
                    np.stack([t.x1, t.y1, t.x2 - t.x1, t.y2 - t.y1], 1)
                    if len(t) else np.empty((0, 4)),
                    max_iou=0.5)
                # ids must be numeric for motmetrics; per-block accumulation
                # already makes the raw id unique, so gid is not needed here
                acc.update(g.id.astype(int).tolist(), t.track_id.tolist(), dist, frameid=f)
            accs.append(acc); names.append(bid)
            ev = acc.mot_events
            sw = ev[ev.Type == "SWITCH"]
            switch_frames += [(bid, int(f)) for f, _ in sw.index]

        summary = mh.compute_many(accs, metrics=want, names=names, generate_overall=True)
        metrics["tracking"] = {n: summary.loc[n].to_dict() for n in summary.index}
        for n in summary.index:
            r = summary.loc[n]
            log.info("  %-8s MOTA=%7.3f IDF1=%.3f IDsw=%4d frag=%4d MT=%d ML=%d objs=%d",
                     n, r.mota, r.idf1, int(r.num_switches), int(r.num_fragmentations),
                     int(r.mostly_tracked), int(r.mostly_lost), int(r.num_unique_objects))

        # ---- where do ID switches happen? --------------------------------
        # Rate per annotated frame, so contest and non-contest are comparable
        # even though the windows differ hugely in length.
        sk = resolve(cfg, "gt") / args.video_id / "gt_events_skeleton.csv"
        pad = float(cfg["annotation"]["contest_pad_s"])
        ctx = {}
        if sk.exists() and switch_frames:
            peaks = pd.read_csv(sk).t_peak_s_approx.dropna().values
            sf = np.array([f for _, f in switch_frames])
            af = np.array(frames)

            def near(fr, centres, pad_s):
                if not len(centres):
                    return np.zeros(len(fr), bool)
                d = np.abs(fr[:, None] / fps - np.asarray(centres)[None, :])
                return (d <= pad_s).any(axis=1)

            in_c, in_c_all = near(sf, peaks, pad), near(af, peaks, pad)
            ctx["contest"] = {
                "pad_s": pad, "n_switches": int(in_c.sum()),
                "n_annotated_frames": int(in_c_all.sum()),
                "switches_per_frame": float(in_c.sum() / max(in_c_all.sum(), 1))}
            ctx["elsewhere"] = {
                "n_switches": int((~in_c).sum()),
                "n_annotated_frames": int((~in_c_all).sum()),
                "switches_per_frame": float((~in_c).sum() / max((~in_c_all).sum(), 1))}
            log.info("  ID switches at contests (+/-%.0fs): %d over %d frames (%.3f/frame) "
                     "vs %d over %d elsewhere (%.3f/frame)",
                     pad, ctx["contest"]["n_switches"], ctx["contest"]["n_annotated_frames"],
                     ctx["contest"]["switches_per_frame"], ctx["elsewhere"]["n_switches"],
                     ctx["elsewhere"]["n_annotated_frames"],
                     ctx["elsewhere"]["switches_per_frame"])

            # Hard cuts and fast pans are separate mechanisms with separate
            # signatures, and docs/10 `camera_pan_cut` bundles them. Both are
                # adjudicated by hand, so both are reported separately or not at all.
            blk = cfg["annotation"]["blocks"]
            for kind, key in [("cut", "confirmed_cuts_s"), ("pan", "confirmed_pans_s")]:
                times = blk.get(key) or []
                if not times:
                    log.info("  annotation.blocks.%s is empty — no %s breakdown. "
                             "This is a HUMAN verdict; it is never inferred from a "
                             "threshold here.", key, kind)
                    continue
                # Restrict to the block(s) actually containing these times, so the
                # comparison is within-block. Pooling across blocks would compare a
                # 2 s window in one block against 58 s of a different scene.
                blocks_hit = sorted({b for b, gb in gt_tr.groupby("block_id")
                                     if any(gb.frame_idx.min() / fps <= t <= gb.frame_idx.max() / fps
                                            for t in times)})
                bf = np.array([f for b, f in switch_frames if b in blocks_hit])
                ba = np.array([f for f in frames
                               if any(gt_tr[(gt_tr.block_id == b)].frame_idx.min() <= f
                                      <= gt_tr[(gt_tr.block_id == b)].frame_idx.max()
                                      for b in blocks_hit)])
                if not len(bf) or not len(ba):
                    log.warning("  no switches or frames found in %s for the %s breakdown",
                                blocks_hit, kind)
                    continue
                inw, inw_all = near(bf, times, 1.0), near(ba, times, 1.0)
                ctx[f"confirmed_{kind}"] = {
                    f"{kind}s_s": list(times), "pad_s": 1.0, "blocks": blocks_hit,
                    "n_switches": int(inw.sum()),
                    "n_annotated_frames": int(inw_all.sum()),
                    "switches_per_frame": float(inw.sum() / max(inw_all.sum(), 1))}
                ctx[f"away_from_{kind}"] = {
                    "blocks": blocks_hit,
                    "n_switches": int((~inw).sum()),
                    "n_annotated_frames": int((~inw_all).sum()),
                    "switches_per_frame": float((~inw).sum() / max((~inw_all).sum(), 1))}
                a, b = ctx[f"confirmed_{kind}"], ctx[f"away_from_{kind}"]
                log.info("  ID switches within +/-1s of a confirmed %s %s: %d over %d frames "
                         "(%.3f/frame) vs %d over %d elsewhere in %s (%.3f/frame) -> %.2fx",
                         kind, list(times), a["n_switches"], a["n_annotated_frames"],
                         a["switches_per_frame"], b["n_switches"], b["n_annotated_frames"],
                         blocks_hit, b["switches_per_frame"],
                         a["switches_per_frame"] / max(b["switches_per_frame"], 1e-9))
        metrics["switch_context"] = ctx

        # ---- GT vs predicted fragmentation -------------------------------
        gt_ids = gt_tr.groupby("block_id").gid.nunique().to_dict()
        pr_ids = {b: int(tr[tr.frame_idx.isin(gt_tr[gt_tr.block_id == b].frame_idx)]
                         .track_id.nunique()) for b in gt_ids}
        metrics["identity_inflation"] = {
            b: {"gt_ids": int(gt_ids[b]), "pred_ids": pr_ids[b],
                "ratio": pr_ids[b] / max(gt_ids[b], 1)} for b in gt_ids}
        for b in gt_ids:
            log.info("  %s: %d GT identities vs %d predicted (%.1fx inflation)",
                     b, gt_ids[b], pr_ids[b], pr_ids[b] / max(gt_ids[b], 1))
    except Exception as e:      # noqa: BLE001
        log.warning("tracking metrics unavailable (%s) — report detection only and say so", e)
        metrics["tracking"] = {"error": str(e)}

    (od / "metrics_detection.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("P=%s", res["precision"]); log.info("R=%s", res["recall"]); log.info("F1=%s", res["f1"])
    log.info("recall small=%.3f medium=%.3f large=%.3f — the drop is the finding",
             *[metrics["recall_by_size"][s]["recall"] for s in ["small", "medium", "large"]])


if __name__ == "__main__":
    main()
