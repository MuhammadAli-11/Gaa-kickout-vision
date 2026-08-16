"""Temporal block cross-validation with a purge gap.

Random k-fold on overlapping sliding windows leaks: window t and window
t+1 share 83% of their frames at a 3 s window / 0.5 s stride. Blocked
splits with a gap are the minimum defensible design, and being able to
say why is worth more in the interview than the classifier itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import temporal_iou


def label_windows(W: pd.DataFrame, gt: pd.DataFrame, tiou_thresh: float = 0.5) -> np.ndarray:
    M = temporal_iou(W[["w_start_s", "w_end_s"]].values, gt[["t_start_s", "t_end_s"]].values)
    return (M.max(axis=1) >= tiou_thresh).astype(int) if M.size else np.zeros(len(W), int)


def temporal_block_cv(X, y, t_start, model, n_splits: int = 5, gap_s: float = 5.0) -> np.ndarray:
    """Out-of-fold scores. Contiguous time blocks, purged either side."""
    order = np.argsort(t_start)
    blocks = np.array_split(order, n_splits)
    oof = np.zeros(len(y), dtype=float)
    for b in blocks:
        lo, hi = t_start[b].min() - gap_s, t_start[b].max() + gap_s
        train = np.where((t_start < lo) | (t_start > hi))[0]
        if len(np.unique(y[train])) < 2:
            oof[b] = float(np.mean(y))       # degenerate fold: report base rate, do not pretend
            continue
        model.fit(X[train], y[train])
        oof[b] = model.predict_proba(X[b])[:, 1]
    return oof
