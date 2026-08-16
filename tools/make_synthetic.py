#!/usr/bin/env python3
"""Synthetic kickout scenarios with a KNOWN ground truth, in pitch space.

Why this exists
---------------
The evaluation half of this pipeline can be tested, demonstrated and
debugged before a single frame of footage is annotated — and, more
usefully, it can be tested against data whose correct answer is known by
construction. The generator places players at chosen pitch coordinates,
picks a defensive strategy per kickout, projects everything through a
synthetic camera homography, and adds detection noise. The true defender
count inside the 65 is therefore exact, so any disagreement in S11 is the
pipeline's, not the labels'.

That is worth saying out loud in an interview: it separates "the number
came out" from "the number is right".

Usage:
    python tools/make_synthetic.py --n-kickouts 16 --noise 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lib.pitch import Pitch, fit_homography          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["zonal", "player_to_player", "concede"]


def synthetic_camera(pitch: Pitch, rng, W=1280, H=720):
    """A plausible broadcast-camera homography: pitch metres -> image px.

    Built by choosing where four pitch landmarks land in the image, then
    solving. Slightly different per shot, as a real camera would be.
    """
    L, Wd = pitch.length / 2, pitch.width / 2
    jitter = lambda s: rng.normal(0, s, 2)                       # noqa: E731
    src = np.array([[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]], float)
    dst = np.array([
        [60, 640], [1220, 640], [980, 300], [300, 300]], float)
    dst = dst + np.stack([jitter(18) for _ in range(4)])
    H_p2i, _ = fit_homography(src, dst)          # note: maps pitch -> image
    H_i2p = np.linalg.inv(H_p2i)
    return H_p2i, H_i2p


def place_kickout(pitch: Pitch, strategy: str, rng) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (defender_xy, attacker_xy, true_count_inside_65)."""
    line65 = pitch.line_x(65.0, "attacking")
    L, Wd = pitch.length / 2, pitch.width / 2

    n_in = {"zonal": rng.integers(8, 13), "player_to_player": rng.integers(7, 12),
            "concede": rng.integers(0, 5)}[strategy]
    n_def = 15
    # attackers spread through the kickout drop zone
    n_att = 13
    att = np.stack([rng.uniform(line65 - 5, L - 25, n_att),
                    rng.uniform(-Wd + 6, Wd - 6, n_att)], 1)

    if strategy == "player_to_player":
        # defenders sit ~1 m off an attacker: the paper's definition
        idx = rng.choice(n_att, min(n_in, n_att), replace=False)
        near = att[idx] + rng.normal(0, 0.55, (len(idx), 2))
        rest = np.stack([rng.uniform(-L + 20, line65 - 8, n_def - len(near)),
                         rng.uniform(-Wd + 5, Wd - 5, n_def - len(near))], 1)
        dfd = np.vstack([near, rest])
    else:
        # zonal / concede: defenders occupy space, not opponents
        near = np.stack([rng.uniform(line65 + 2, L - 20, n_in),
                         rng.uniform(-Wd + 8, Wd - 8, n_in)], 1)
        rest = np.stack([rng.uniform(-L + 20, line65 - 8, n_def - n_in),
                         rng.uniform(-Wd + 5, Wd - 5, n_def - n_in)], 1)
        dfd = np.vstack([near, rest])

    # the kicking team's goalkeeper: deep in their own half, well behind
    # everyone else. Without one, the defending team cannot be identified.
    keeper = np.array([[-L + 4.0, rng.uniform(-3, 3)]])
    att = np.vstack([att, keeper])

    true_count = int((dfd[:, 0] >= line65).sum())
    return dfd, att, true_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="synthetic_demo")
    ap.add_argument("--n-kickouts", type=int, default=16)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--gap-s", type=float, default=35.0)
    ap.add_argument("--noise", type=float, default=0.4,
                    help="0 = perfect detection/registration, 1 = hopeless")
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pitch = Pitch()
    out = ROOT / "outputs" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    gt_dir = ROOT / "data" / "gt" / args.video_id
    gt_dir.mkdir(parents=True, exist_ok=True)

    setup_s, hold_s = 1.5, 2.0
    rows, shots, Hs, gt_ev, gt_cod = [], [], {}, [], []
    frame = 0

    for k in range(args.n_kickouts):
        strategy = STRATEGIES[rng.choice(3, p=[0.53, 0.43, 0.04])]
        dfd, att, true_count = place_kickout(pitch, strategy, rng)
        H_p2i, H_i2p = synthetic_camera(pitch, rng)
        Hs[str(k)] = H_i2p.tolist()

        t0 = k * args.gap_s + 10.0
        f_start = int((t0 - setup_s) * args.fps)
        f_end = int((t0 + hold_s) * args.fps)
        shots.append(dict(shot_id=k, frame_start=f_start, frame_end=f_end,
                          t_start_s=f_start / args.fps, t_end_s=f_end / args.fps,
                          keyframe=(f_start + f_end) // 2))

        players = np.vstack([dfd, att])
        team = ["team_b"] * len(dfd) + ["team_a"] * len(att)
        for f in range(f_start, f_end + 1):
            drift = rng.normal(0, 0.25 * args.noise, players.shape)
            pos = players + drift
            hom = np.concatenate([pos, np.ones((len(pos), 1))], 1) @ H_p2i.T
            img = hom[:, :2] / hom[:, 2:3]
            img += rng.normal(0, 6 * args.noise, img.shape)      # detection jitter
            keep = rng.random(len(pos)) > 0.05 * (1 + args.noise)
            for i in np.where(keep)[0]:
                h = 46.0 + rng.normal(0, 2)
                rows.append((k * 100 + i, f, f / args.fps,
                             img[i, 0] - h * 0.22, img[i, 1] - h, img[i, 0] + h * 0.22, img[i, 1],
                             img[i, 0], img[i, 1] - h / 2, h, h * h * 0.44,
                             float(rng.uniform(0.4, 0.95)), team[i]))
        frame = f_end

        gt_ev.append(dict(event_id=f"ko_{k:03d}", t_start_s=t0, t_end_s=t0 + 4.0,
                          t_peak_s=t0 + 1.8, n_players_in_contest=int(rng.integers(2, 7)),
                          outcome="own_clean", restart_type="long", coder_id="synthetic",
                          coding_pass=1, visibility="full", notes=""))
        band = next(lbl for lo, hi, lbl in [(0, 4, "0-4"), (5, 7, "5-7"), (8, 10, "8-10"),
                                            (11, 99, "11+")] if lo <= true_count <= hi)
        gt_cod.append(dict(event_id=f"ko_{k:03d}", n_defenders_in_65=true_count,
                           n_defenders_band=band, defensive_strategy=strategy,
                           distance_class="long", contested_proxy="contested",
                           kickout_won="won", coder_id="synthetic", coding_pass=1,
                           confidence="high", notes="ground truth by construction"))

    tr = pd.DataFrame(rows, columns=["track_id", "frame_idx", "timestamp_s", "x1", "y1", "x2", "y2",
                                     "cx", "cy", "bbox_h", "bbox_area", "conf", "team"])
    teams = tr[["track_id", "team"]].drop_duplicates()
    # a realistic slice of tracks cannot be assigned a team
    n_un = int(len(teams) * 0.10 * (1 + args.noise))
    un = rng.choice(teams.track_id.values, min(n_un, len(teams)), replace=False)
    teams["confidence"] = rng.uniform(0.15, 0.9, len(teams))
    teams.loc[teams.track_id.isin(un), ["team", "confidence"]] = ["unassigned", 0.05]
    teams["n_crops"] = 25
    teams["method"] = "hsv_torso_kmeans"

    for c in ["x1", "y1", "x2", "y2", "cx", "cy", "bbox_h", "bbox_area", "conf"]:
        tr[c] = tr[c].astype("float32")
    tr.drop(columns=["team"]).to_parquet(out / "tracks.parquet", index=False)
    teams.to_csv(out / "track_teams.csv", index=False)
    pd.DataFrame(shots).to_csv(out / "shots.csv", index=False)
    (out / "homographies.json").write_text(json.dumps(Hs, indent=2))

    gt_ev = pd.DataFrame(gt_ev)
    gt_cod = pd.DataFrame(gt_cod)
    # blind re-code of a random 25%, with realistic human inconsistency
    sub = gt_cod.sample(frac=0.25, random_state=args.seed).copy()
    sub["n_defenders_in_65"] = (sub.n_defenders_in_65 + rng.integers(-1, 2, len(sub))).clip(0, 15)
    sub["coding_pass"] = 2
    gt_cod = pd.concat([gt_cod, sub], ignore_index=True)

    gt_ev.to_csv(gt_dir / "gt_events.csv", index=False)
    gt_cod.to_csv(gt_dir / "gt_coding.csv", index=False)

    print(f"synthetic: {args.n_kickouts} kickouts, {len(tr)} track rows, "
          f"{len(teams)} tracks ({int((teams.team=='unassigned').sum())} unassigned)")
    print(f"  strategies: {gt_cod[gt_cod.coding_pass==1].defensive_strategy.value_counts().to_dict()}")
    print(f"  true defender counts inside the 65: "
          f"{gt_cod[gt_cod.coding_pass==1].n_defenders_in_65.tolist()}")
    print(f"  -> {out}  and  {gt_dir}")


if __name__ == "__main__":
    main()
