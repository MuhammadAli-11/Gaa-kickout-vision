#!/usr/bin/env python3
"""S04 — Pitch registration. Map image pixels to pitch metres.

Why this stage exists
---------------------
Every variable in McColgan et al. (2026) is defined in pitch space:
defenders inside the opposition 65, kickouts received inside or outside
the 45, an opponent within 2 m of the receiver, four players across the
21. Replicating that coding scheme from video is therefore a
REGISTRATION problem before it is a detection problem, and a project
that works in pixels cannot produce their variables at all.

Approach
--------
One homography per camera SHOT, not per frame. Broadcast cameras pan and
zoom within a shot, so a single homography per shot is an approximation;
its cost is measured (drift, below) rather than assumed away. Shots come
from the scene-change flag, or from a shot list you supply.

Landmarks are clicked by hand on one keyframe per shot. This is
semi-automatic and honest about it: automatic pitch-line registration on
broadcast GAA footage is a research problem in its own right, and
pretending to solve it in a week would be the wrong claim to make.

Outputs the reprojection error in METRES on held-out landmarks. That
number is the floor on everything downstream: if it is 3 m, then "inside
the 65" is a coin-flip for any player within 3 m of the line, and the
count variable inherits that uncertainty.

Usage
-----
    # 1. list shots to register
    python src/s04_register_pitch.py --video-id <id> --list-shots

    # 2. click landmarks on each shot keyframe (writes pitch_keypoints.csv)
    python src/s04_register_pitch.py --video-id <id> --annotate --shot 3

    # 3. fit and validate all homographies
    python src/s04_register_pitch.py --video-id <id> --fit
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir, make_provenance
from lib.logging_setup import get_logger
from lib.pitch import Pitch, fit_homography, apply_homography, reprojection_error_m
from lib.schema import read_table, write_table


def shots_from_features(feat: pd.DataFrame, min_len_frames: int = 25) -> pd.DataFrame:
    """Segment the clip into camera shots using the scene-change flag."""
    cuts = [0] + feat.index[feat.scene_change_flag == 1].tolist() + [len(feat)]
    rows, prev = [], 0
    for c in cuts[1:]:
        if c - prev >= min_len_frames:
            rows.append(dict(shot_id=len(rows), frame_start=int(prev), frame_end=int(c) - 1,
                             t_start_s=float(feat.timestamp_s.iloc[prev]),
                             t_end_s=float(feat.timestamp_s.iloc[min(c - 1, len(feat) - 1)]),
                             keyframe=int((prev + c) // 2)))
        prev = c
    return pd.DataFrame(rows)


def annotate(cfg: dict, video_id: str, shot: pd.Series, pitch: Pitch, log) -> pd.DataFrame:
    """Click landmarks on a keyframe. Left-click a point, then type its name."""
    import cv2
    frames_dir = resolve(cfg, "frames") / video_id
    img_path = frames_dir / f"{int(shot.keyframe):06d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"no keyframe at {img_path} — run s01 without --skip-frames")

    names = list(pitch.landmarks())
    print("\nLandmarks available (click, then enter the number):")
    for i, n in enumerate(names):
        print(f"  {i:2d} {n}")
    print("\nClick a point, then type its number in the terminal. 'q' in the window when done.\n")

    clicks: list[tuple[float, float]] = []
    disp = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y)))
            cv2.circle(disp, (x, y), 4, (0, 220, 120), -1)
            cv2.imshow("register", disp)

    cv2.imshow("register", disp)
    cv2.setMouseCallback("register", on_click)
    rows = []
    while True:
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
        if len(clicks) > len(rows):
            idx = input(f"  landmark for click {len(rows) + 1} at {clicks[-1]}: ").strip()
            if idx.isdigit() and int(idx) < len(names):
                nm = names[int(idx)]
                px, py = pitch.landmarks()[nm]
                rows.append(dict(video_id=video_id, shot_id=int(shot.shot_id),
                                 frame_idx=int(shot.keyframe), landmark=nm,
                                 img_x=clicks[-1][0], img_y=clicks[-1][1],
                                 pitch_x=px, pitch_y=py, use_for_fit=1))
            else:
                clicks.pop()
    cv2.destroyAllWindows()
    log.info("shot %d: %d landmarks clicked", shot.shot_id, len(rows))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--list-shots", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--shot", type=int)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--holdout", type=int, default=2, help="landmarks held out per shot for validation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s04_register", od / "logs")
    pitch = Pitch(**cfg["pitch"]["dimensions"])

    feat_path = od / "features_image.parquet"
    if not feat_path.exists():
        raise SystemExit("run s03_track then s05 --image-only first, or supply a shot list")
    feat = read_table("features_image", feat_path)
    shots = shots_from_features(feat, cfg["pitch"]["min_shot_len_frames"])
    shots_path = od / "shots.csv"
    shots.to_csv(shots_path, index=False)

    if args.list_shots:
        log.info("%d shots >= %d frames", len(shots), cfg["pitch"]["min_shot_len_frames"])
        print(shots.to_string(index=False))
        log.info("Register only the shots that CONTAIN a kickout. Registering all of "
                 "them is usually wasted work — see docs/03 §D.")
        return

    kp_path = resolve(cfg, "gt") / args.video_id / "pitch_keypoints.csv"

    if args.annotate:
        if args.shot is None:
            raise SystemExit("--annotate needs --shot N")
        new = annotate(cfg, args.video_id, shots.iloc[args.shot], pitch, log)
        old = pd.read_csv(kp_path) if kp_path.exists() else pd.DataFrame()
        keep = old[old.shot_id != args.shot] if len(old) else old
        write_table(pd.concat([keep, new], ignore_index=True), "pitch_keypoints", kp_path)
        log.info("saved -> %s", kp_path)
        return

    if not args.fit:
        raise SystemExit("choose one of --list-shots / --annotate / --fit")

    # ---------------- fit + validate -------------------------------
    kp = read_table("pitch_keypoints", kp_path)
    rows, Hs = [], {}
    rng = np.random.default_rng(cfg["project"]["seed"])
    for shot_id, g in kp.groupby("shot_id"):
        g = g[g.use_for_fit == 1]
        if len(g) < 4 + args.holdout:
            log.warning("shot %d: only %d landmarks — need >= %d for a validated fit; "
                        "fitting without hold-out and flagging it", shot_id, len(g), 4 + args.holdout)
            hold = np.zeros(len(g), bool)
        else:
            hold = np.zeros(len(g), bool)
            hold[rng.choice(len(g), args.holdout, replace=False)] = True

        fit_g, val_g = g[~hold], g[hold]
        H, inliers = fit_homography(fit_g[["img_x", "img_y"]].values,
                                    fit_g[["pitch_x", "pitch_y"]].values,
                                    cfg["pitch"]["ransac_px"])
        e_fit = reprojection_error_m(H, fit_g[["img_x", "img_y"]].values,
                                     fit_g[["pitch_x", "pitch_y"]].values)
        e_val = (reprojection_error_m(H, val_g[["img_x", "img_y"]].values,
                                      val_g[["pitch_x", "pitch_y"]].values)
                 if len(val_g) else {"rmse_m": np.nan, "median_m": np.nan, "max_m": np.nan, "n": 0})
        Hs[int(shot_id)] = H.tolist()
        rows.append(dict(shot_id=int(shot_id), n_landmarks=int(len(g)),
                         n_inliers=int(inliers.sum()),
                         rmse_fit_m=e_fit["rmse_m"], rmse_holdout_m=e_val["rmse_m"],
                         max_holdout_m=e_val["max_m"], n_holdout=e_val["n"],
                         validated=int(e_val["n"] > 0)))
        log.info("shot %2d | n=%2d | RMSE fit %.2f m | RMSE hold-out %.2f m",
                 shot_id, len(g), e_fit["rmse_m"], e_val["rmse_m"])

    hom = pd.DataFrame(rows)
    write_table(hom, "homographies", od / "homographies.csv")
    (od / "homographies.json").write_text(json.dumps(Hs, indent=2))

    val = hom[hom.validated == 1]
    if len(val):
        med = float(np.nanmedian(val.rmse_holdout_m))
        log.info("median held-out reprojection error: %.2f m across %d shots", med, len(val))
        log.info("INTERPRET: this is the floor on every pitch-space variable. A player "
                 "standing within %.1f m of the 65 m line cannot be classified reliably, "
                 "so report the count variable with that ambiguity band.", med)
        if med > cfg["pitch"]["max_acceptable_rmse_m"]:
            log.warning("error exceeds the %.1f m threshold in config. Either add landmarks "
                        "on this shot or exclude it — do not code kickouts from a shot whose "
                        "registration you do not trust.", cfg["pitch"]["max_acceptable_rmse_m"])

    make_provenance("s04_register", cfg, [str(kp_path)],
                    dict(n_shots=len(hom), **cfg["pitch"])).write(od / "provenance_s04.json")


if __name__ == "__main__":
    main()
