"""Frame <-> time conversion. The single place this arithmetic happens.

Rule for this project: after s01, the working clip has a CONSTANT frame
rate, frame_idx is 0-based and contiguous, and

        timestamp_s = frame_idx / fps

is exact by construction. Never read timestamps back out of the
container (VFR broadcast sources lie), never use cv2.CAP_PROP_POS_MSEC.
"""
from __future__ import annotations

import numpy as np


def frame_to_time(frame_idx, fps: float):
    return np.asarray(frame_idx, dtype=np.float64) / float(fps)


def time_to_frame(t_s, fps: float):
    return np.rint(np.asarray(t_s, dtype=np.float64) * float(fps)).astype(np.int64)


def hhmmss_to_seconds(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def seconds_to_hhmmss(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
