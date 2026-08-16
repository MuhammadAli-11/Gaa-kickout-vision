#!/usr/bin/env python3
"""Select random time blocks for the blind re-code (docs/03 §C).

Blocks, not events: sampling 25% of events cannot reveal an event you
missed entirely on pass 1, and missed events are exactly what the
intra-rater check is supposed to catch.
"""
from __future__ import annotations

import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, required=True, help="clip length in seconds")
ap.add_argument("--fraction", type=float, default=0.25)
ap.add_argument("--block-s", type=float, default=60.0)
ap.add_argument("--seed", type=int, default=1729)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
n_blocks = int(np.ceil(a.duration / a.block_s))
k = max(1, int(round(n_blocks * a.fraction)))
chosen = sorted(rng.choice(n_blocks, size=k, replace=False))

print(f"Re-code these {k} of {n_blocks} blocks ({a.block_s:.0f}s each), seed={a.seed}:\n")
for b in chosen:
    print(f"  block {b:2d}:  {b * a.block_s:7.1f}s  to  {min((b + 1) * a.block_s, a.duration):7.1f}s")
print("\nDo NOT look at pass 1 while coding these.")
