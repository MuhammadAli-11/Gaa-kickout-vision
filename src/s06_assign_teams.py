#!/usr/bin/env python3
"""S06 — Team assignment. The dependency that unlocks the paper's variables.

Why this is here
----------------
Almost every variable in McColgan et al. (2026) is asymmetric between
the two teams: players *the defending team* commits inside the
opposition 65; whether *defenders* stay within a metre of *offensive*
players; whether the *kicking* team retains possession. A detector that
returns undifferentiated "person" boxes cannot express any of that.
Team assignment is therefore not a nicety, it is the blocking dependency
for reproducing their coding scheme, and a project that skips it can
only ever produce proxies.

Method
------
Torso-crop colour histogram in HSV, per track, aggregated over the
track's lifetime (a single frame is too noisy), then k-means into two
clusters with outliers held aside for goalkeepers and officials. This is
the standard cheap approach. It is also the one that fails on kit clashes,
in shadow, under floodlights, and on the two or three tracks per clip that
are half-occluded throughout — so this stage reports its own confidence
and marks low-confidence tracks `unassigned` rather than guessing.

An unassigned track is not a failure to hide. Propagate it: a kickout
where 4 of 22 players are unassigned has a defender count with an
uncertainty of ±4, and that band belongs in the result.

Usage:
    python src/s06_assign_teams.py --video-id <id>
    python src/s06_assign_teams.py --video-id <id> --review   # fix by hand
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir, set_seeds, make_provenance
from lib.logging_setup import get_logger
from lib.schema import read_table, write_table


def torso_hist(img, box, bins=(8, 8)) -> np.ndarray | None:
    """HS histogram of the torso region (upper-middle of the box)."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = img.shape[:2]
    bh, bw = y2 - y1, x2 - x1
    if bh < 12 or bw < 6:
        return None                       # too small to carry colour information
    ty1 = max(y1 + int(0.18 * bh), 0)
    ty2 = min(y1 + int(0.55 * bh), h)
    tx1 = max(x1 + int(0.20 * bw), 0)
    tx2 = min(x2 - int(0.20 * bw), w)
    if ty2 <= ty1 or tx2 <= tx1:
        return None
    crop = img[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)).astype(np.uint8)
    if mask.sum() < 20:
        return None                       # washed out: grass glare or deep shadow
    hist = cv2.calcHist([hsv], [0, 1], mask, list(bins), [0, 180, 0, 256])
    return (hist / (hist.sum() + 1e-9)).flatten()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--sample-per-track", type=int, default=25)
    ap.add_argument("--review", action="store_true",
                    help="print low-confidence tracks for manual correction")
    args = ap.parse_args()

    import cv2
    from sklearn.cluster import KMeans

    cfg = load_config(args.config)
    set_seeds(cfg)
    od = out_dir(cfg)
    log = get_logger("s06_teams", od / "logs")
    T = cfg["teams"]

    tr = read_table("tracks", od / "tracks.parquet")
    clip = resolve(cfg, "clips") / f"{args.video_id}_working.mp4"
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")

    # sample frames once, in order — seeking per track would be far slower
    rng = np.random.default_rng(cfg["project"]["seed"])
    sample = (tr.groupby("track_id", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.sample_per_track), random_state=1729))
                .sort_values("frame_idx"))
    log.info("sampling %d crops across %d tracks", len(sample), tr.track_id.nunique())

    feats: dict[int, list[np.ndarray]] = {}
    cur_frame, img = -1, None
    for r in sample.itertuples():
        if r.frame_idx != cur_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame_idx))
            ok, img = cap.read()
            cur_frame = r.frame_idx
            if not ok:
                continue
        h = torso_hist(img, (r.x1, r.y1, r.x2, r.y2))
        if h is not None:
            feats.setdefault(int(r.track_id), []).append(h)
    cap.release()

    ids = sorted(feats)
    X = np.stack([np.mean(feats[i], axis=0) for i in ids])
    n_used = np.array([len(feats[i]) for i in ids])
    log.info("usable colour signatures for %d of %d tracks", len(ids), tr.track_id.nunique())

    km = KMeans(n_clusters=2, n_init=20, random_state=cfg["project"]["seed"]).fit(X)
    d = km.transform(X)
    margin = np.abs(d[:, 0] - d[:, 1]) / (d.sum(axis=1) + 1e-9)   # 0 = ambiguous, 1 = clean

    team = np.where(km.labels_ == 0, "team_a", "team_b").astype(object)
    low = margin < T["min_margin"]
    team[low] = "unassigned"
    team[n_used < T["min_crops"]] = "unassigned"

    out = pd.DataFrame(dict(
        track_id=ids, team=team, confidence=margin.astype(float),
        n_crops=n_used, method="hsv_torso_kmeans",
    ))
    missing = sorted(set(tr.track_id.unique()) - set(ids))
    if missing:
        out = pd.concat([out, pd.DataFrame(dict(
            track_id=missing, team="unassigned", confidence=0.0, n_crops=0,
            method="no_usable_crop"))], ignore_index=True)

    write_table(out, "track_teams", od / "track_teams.csv")

    frac_unassigned = float((out.team == "unassigned").mean())
    log.info("team_a=%d  team_b=%d  unassigned=%d (%.0f%%)",
             int((out.team == "team_a").sum()), int((out.team == "team_b").sum()),
             int((out.team == "unassigned").sum()), 100 * frac_unassigned)
    log.info("PROPAGATE THIS: every count variable inherits an uncertainty of "
             "+/- the number of unassigned players present. Report the band, not just "
             "the point count.")
    if frac_unassigned > T["max_unassigned_frac"]:
        log.warning("more than %.0f%% unassigned. Kit colours may clash, or the clip may be "
                    "too distant for torso colour. Options: hand-label teams for the "
                    "kickout frames only (docs/03 §E), or report the count variable as "
                    "TOTAL players inside the 65 rather than by team, and say why.",
                    100 * T["max_unassigned_frac"])

    if args.review:
        print("\nLowest-confidence tracks — check these first:\n")
        print(out.nsmallest(15, "confidence").to_string(index=False))
        print("\nEdit track_teams.csv by hand and set method='manual' for rows you fix.")

    make_provenance("s06_teams", cfg, ["tracks.parquet"],
                    dict(**T, frac_unassigned=frac_unassigned)).write(od / "provenance_s06.json")


if __name__ == "__main__":
    main()
