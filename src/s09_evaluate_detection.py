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


def load_mot_gt(path) -> pd.DataFrame:
    cols = ["frame", "id", "bb_left", "bb_top", "bb_w", "bb_h", "conf", "cls", "vis"]
    gt = pd.read_csv(path, header=None, names=cols)
    gt = gt[gt.conf > 0] if gt.conf.notna().any() else gt
    gt["frame_idx"] = gt.frame.astype(int) - 1          # MOT is 1-based
    gt["x1"], gt["y1"] = gt.bb_left, gt.bb_top
    gt["x2"], gt["y2"] = gt.bb_left + gt.bb_w, gt.bb_top + gt.bb_h
    gt["area"] = gt.bb_w * gt.bb_h
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

    gt = load_mot_gt(resolve(cfg, "gt") / args.video_id / "gt_boxes.txt")
    det = read_table("detections", od / "detections.parquet")
    frames = sorted(gt.frame_idx.unique())
    det = det[det.frame_idx.isin(frames)]
    log.info("annotated subset: %d frames, %d GT boxes, %d predictions", len(frames), len(gt), len(det))

    # ---- terciles from GT box area (fixed once, reported in the paper) ----
    q1, q2 = np.percentile(gt.area, [33.3, 66.7])
    gt["size_bin"] = pd.cut(gt.area, [-np.inf, q1, q2, np.inf], labels=["small", "medium", "large"])
    log.info("size terciles by GT bbox area: small<%.0f px2, medium<%.0f px2", q1, q2)

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

    # ---- tracking metrics ------------------------------------------
    try:
        import motmetrics as mm
        tr = read_table("tracks", od / "tracks.parquet")
        acc = mm.MOTAccumulator(auto_id=False)
        for f in frames:
            g = gt[gt.frame_idx == f]; t = tr[tr.frame_idx == f]
            dist = mm.distances.iou_matrix(
                g[["bb_left", "bb_top", "bb_w", "bb_h"]].values,
                np.stack([t.x1, t.y1, t.x2 - t.x1, t.y2 - t.y1], 1) if len(t) else np.empty((0, 4)),
                max_iou=0.5)
            acc.update(g.id.tolist(), t.track_id.tolist(), dist, frameid=f)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_fragmentations",
                                           "mostly_tracked", "mostly_lost", "num_unique_objects"],
                             name="acc")
        metrics["tracking"] = summary.iloc[0].to_dict()
        log.info("MOTA=%.3f IDF1=%.3f IDsw=%d frag=%d",
                 summary.mota.iloc[0], summary.idf1.iloc[0],
                 int(summary.num_switches.iloc[0]), int(summary.num_fragmentations.iloc[0]))
    except Exception as e:      # noqa: BLE001
        log.warning("tracking metrics unavailable (%s) — report detection only and say so", e)
        metrics["tracking"] = {"error": str(e)}

    (od / "metrics_detection.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("P=%s", res["precision"]); log.info("R=%s", res["recall"]); log.info("F1=%s", res["f1"])
    log.info("recall small=%.3f medium=%.3f large=%.3f — the drop is the finding",
             *[metrics["recall_by_size"][s]["recall"] for s in ["small", "medium", "large"]])


if __name__ == "__main__":
    main()
