"""Matching utilities: spatial IoU for boxes, temporal IoU for events.

Event matching is greedy-by-confidence with one-to-one constraint, which
is the standard used in temporal action localisation. Doing it any other
way (e.g. nearest-neighbour without the one-to-one constraint) inflates
recall, and an examiner will ask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def temporal_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred: (N,2) [t_start,t_end], gt: (M,2) -> (N,M) temporal IoU."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    inter = np.clip(
        np.minimum(pred[:, None, 1], gt[None, :, 1]) - np.maximum(pred[:, None, 0], gt[None, :, 0]),
        0, None,
    )
    union = (pred[:, 1] - pred[:, 0])[:, None] + (gt[:, 1] - gt[:, 0])[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_events(pred_df: pd.DataFrame, gt_df: pd.DataFrame, tiou_thresh: float = 0.5) -> pd.DataFrame:
    """Greedy one-to-one matching, highest-confidence prediction first.

    Returns a long table with one row per prediction and per unmatched GT,
    which is what every downstream metric and the failure audit consume.
    """
    pred = pred_df.sort_values("confidence", ascending=False).reset_index(drop=True)
    gt = gt_df.reset_index(drop=True)
    M = temporal_iou(pred[["t_start_s", "t_end_s"]].values, gt[["t_start_s", "t_end_s"]].values)

    taken: set[int] = set()
    rows = []
    for i in range(len(pred)):
        order = np.argsort(-M[i]) if M.shape[1] else []
        j_best, iou_best = -1, 0.0
        for j in order:
            if j in taken:
                continue
            if M[i, j] >= tiou_thresh:
                j_best, iou_best = int(j), float(M[i, j])
            break
        if j_best >= 0:
            taken.add(j_best)
            rows.append(dict(
                kind="TP", pred_id=pred.loc[i, "event_id"], gt_id=gt.loc[j_best, "event_id"],
                confidence=float(pred.loc[i, "confidence"]), tiou=iou_best,
                t_peak_pred=float(pred.loc[i, "t_peak_s"]), t_peak_gt=float(gt.loc[j_best, "t_peak_s"]),
                dur_pred=float(pred.loc[i, "t_end_s"] - pred.loc[i, "t_start_s"]),
                dur_gt=float(gt.loc[j_best, "t_end_s"] - gt.loc[j_best, "t_start_s"]),
                n_pred=pred.loc[i].get("n_players_in_contest", np.nan),
                n_gt=gt.loc[j_best].get("n_players_in_contest", np.nan),
            ))
        else:
            rows.append(dict(
                kind="FP", pred_id=pred.loc[i, "event_id"], gt_id=None,
                confidence=float(pred.loc[i, "confidence"]), tiou=float(M[i].max()) if M.shape[1] else 0.0,
                t_peak_pred=float(pred.loc[i, "t_peak_s"]), t_peak_gt=np.nan,
                dur_pred=float(pred.loc[i, "t_end_s"] - pred.loc[i, "t_start_s"]), dur_gt=np.nan,
                n_pred=pred.loc[i].get("n_players_in_contest", np.nan), n_gt=np.nan,
            ))
    for j in range(len(gt)):
        if j not in taken:
            rows.append(dict(
                kind="FN", pred_id=None, gt_id=gt.loc[j, "event_id"], confidence=np.nan,
                tiou=0.0, t_peak_pred=np.nan, t_peak_gt=float(gt.loc[j, "t_peak_s"]),
                dur_pred=np.nan, dur_gt=float(gt.loc[j, "t_end_s"] - gt.loc[j, "t_start_s"]),
                n_pred=np.nan, n_gt=gt.loc[j].get("n_players_in_contest", np.nan),
            ))
    return pd.DataFrame(rows)


def events_to_bins(df: pd.DataFrame, total_s: float, bin_s: float = 5.0) -> np.ndarray:
    """Binary presence vector over fixed bins — the input to Cohen's kappa."""
    n_bins = int(np.ceil(total_s / bin_s))
    v = np.zeros(n_bins, dtype=int)
    for _, r in df.iterrows():
        lo = int(np.floor(r["t_start_s"] / bin_s))
        hi = int(np.floor(max(r["t_end_s"] - 1e-9, r["t_start_s"]) / bin_s))
        v[max(lo, 0):min(hi + 1, n_bins)] = 1
    return v
