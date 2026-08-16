#!/usr/bin/env python3
"""S05 — Features, in two passes.

Pass 1 (`--image-only`, no homography needed)
    Image-space scene features: player count, spatial dispersion in
    pixels, vertical velocity, camera-cut flag. These drive shot
    segmentation for S04 and temporal kickout localisation in S06.

Pass 2 (default, needs S04 homographies)
    Pitch-space features in METRES, which are what the source paper's
    coding scheme actually requires: how many players are inside the
    opposition 65, how far up the pitch the group sits, how spread it is
    across the pitch, and how fast players move in m/s rather than px/s.

The split matters. Pixel dispersion changes when the camera zooms;
metres do not. Any variable the paper defines by a pitch line has to be
computed after registration, and any variable computed before it is a
proxy whose relationship to the real quantity depends on camera state.

Usage:
    python src/s05_features.py --video-id <id> --image-only   # before S04
    python src/s05_features.py --video-id <id>                # after S04
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib.config import load_config, out_dir, make_provenance
from lib.logging_setup import get_logger
from lib.pitch import Pitch, apply_homography, foot_point
from lib.schema import read_table, write_table
from lib.timebase import frame_to_time


def _mean_knn(px: np.ndarray, py: np.ndarray, k: int) -> float:
    n = len(px)
    if n < 2:
        return np.nan
    p = np.stack([px, py], 1)
    d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.sort(d, axis=1)[:, :min(k, n - 1)].mean())


def _smooth(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, center=True, min_periods=1).mean()


def image_features(tr: pd.DataFrame, cfg: dict, n_frames: int) -> pd.DataFrame:
    f, fps = cfg["features"], cfg["video"]["fps"]
    g = tr.groupby("frame_idx")
    base = pd.DataFrame({
        "n_tracks": g.size(),
        "max_bbox_height": g.bbox_h.max(),
        "mean_bbox_height": g.bbox_h.mean(),
        "sx": g.cx.std(ddof=0),
        "sy": g.cy.std(ddof=0),
    }).reindex(range(n_frames)).fillna({"n_tracks": 0})
    base["cluster_density_px"] = g.apply(
        lambda d: _mean_knn(d.cx.values, d.cy.values, f["n_nearest_for_density"])
    ).reindex(range(n_frames))
    base["centroid_spread_px"] = np.sqrt(base.sx.fillna(0) ** 2 + base.sy.fillna(0) ** 2)

    tr = tr.sort_values(["track_id", "frame_idx"])
    w = f["velocity_window_frames"]
    tr = tr.assign(vy=-tr.groupby("track_id").cy.diff(w) * fps / w)
    vy = tr.groupby("frame_idx").vy.agg(["mean", "max"]).reindex(range(n_frames))
    base["mean_vertical_velocity"] = _smooth(vy["mean"], f["smooth_window_frames"])
    base["max_vertical_velocity"] = _smooth(vy["max"], f["smooth_window_frames"])

    ids = tr.groupby("frame_idx").track_id.apply(set).reindex(range(n_frames))
    churn, prev = np.zeros(n_frames), set()
    for i in range(n_frames):
        cur = ids.iloc[i] if isinstance(ids.iloc[i], set) else set()
        churn[i] = 1.0 - len(prev & cur) / max(len(prev | cur), 1)
        prev = cur
    base["scene_change_flag"] = (churn > f["scene_change_threshold"]).astype("int8")
    base["valid_frame"] = (base.n_tracks >= f["min_tracks_for_valid_frame"]).astype("int8")

    base = base.drop(columns=["sx", "sy"]).reset_index(names="frame_idx")
    base["timestamp_s"] = frame_to_time(base.frame_idx, fps)
    base["n_tracks"] = base.n_tracks.astype("int64")
    for c in base.columns:
        if base[c].dtype == "float64" and c != "timestamp_s":
            base[c] = base[c].astype("float32")
    return base


def pitch_features(tr: pd.DataFrame, cfg: dict, shots: pd.DataFrame, Hs: dict,
                   n_frames: int, log) -> pd.DataFrame:
    """Project every player's ground-contact point and derive metric features."""
    pitch = Pitch(**cfg["pitch"]["dimensions"])
    fps = cfg["video"]["fps"]
    att = cfg["pitch"]["attacking_end"]

    shot_of = np.full(n_frames, -1, dtype=int)
    for r in shots.itertuples():
        shot_of[r.frame_start:r.frame_end + 1] = r.shot_id

    tr = tr.copy()
    tr["shot_id"] = shot_of[tr.frame_idx.values]
    foot = np.array([foot_point(*b) for b in tr[["x1", "y1", "x2", "y2"]].values])
    tr["foot_x"], tr["foot_y"] = foot[:, 0], foot[:, 1]
    tr["pitch_x"], tr["pitch_y"] = np.nan, np.nan

    registered = 0
    for sid, g in tr.groupby("shot_id"):
        H = Hs.get(str(int(sid)))
        if H is None:
            continue
        xy = apply_homography(np.array(H, float), g[["foot_x", "foot_y"]].values)
        tr.loc[g.index, "pitch_x"] = xy[:, 0]
        tr.loc[g.index, "pitch_y"] = xy[:, 1]
        registered += len(g)
    log.info("projected %d of %d track rows (%.0f%%) — unregistered shots yield no "
             "pitch-space variables and must be excluded from coding, not guessed",
             registered, len(tr), 100 * registered / max(len(tr), 1))

    # A projected player outside the pitch is a bad detection or a bad
    # homography. Either way it must not be counted.
    L, W = pitch.length / 2 + 5, pitch.width / 2 + 5
    off = (~tr.pitch_x.between(-L, L)) | (~tr.pitch_y.between(-W, W))
    log.info("%d projections fell outside the pitch envelope and were dropped (%.1f%%)",
             int(off.sum()), 100 * off.mean())
    tr.loc[off, ["pitch_x", "pitch_y"]] = np.nan

    p = tr.dropna(subset=["pitch_x"]).copy()
    line65, line45 = pitch.line_x(65.0, att), pitch.line_x(45.0, att)
    sign = 1 if att == "attacking" else -1
    p["in65"] = (sign * p.pitch_x >= sign * line65).astype(int)
    p["in45"] = (sign * p.pitch_x >= sign * line45).astype(int)

    g = p.groupby("frame_idx")
    out = pd.DataFrame({
        "n_players_registered": g.size(),
        "n_inside_65": g.in65.sum(),
        "n_inside_45": g.in45.sum(),
        "mean_x_m": g.pitch_x.mean(),
        "spread_x_m": g.pitch_x.std(ddof=0),
        "spread_y_m": g.pitch_y.std(ddof=0),
        "width_used_m": g.pitch_y.max() - g.pitch_y.min(),
        "depth_used_m": g.pitch_x.max() - g.pitch_x.min(),
    }).reindex(range(n_frames))
    out["nn_distance_m"] = g.apply(
        lambda d: _mean_knn(d.pitch_x.values, d.pitch_y.values, 1)).reindex(range(n_frames))
    out["cluster_density_m"] = g.apply(
        lambda d: _mean_knn(d.pitch_x.values, d.pitch_y.values,
                            cfg["features"]["n_nearest_for_density"])).reindex(range(n_frames))

    # Metric speed. The paper separates Purposeful from Limited Movement
    # by the INTENT of runs (sprint versus jog). Speed in m/s is the
    # closest observable proxy, and calling it a proxy is the honest move.
    p = p.sort_values(["track_id", "frame_idx"])
    w = cfg["features"]["velocity_window_frames"]
    dx = p.groupby("track_id").pitch_x.diff(w)
    dy = p.groupby("track_id").pitch_y.diff(w)
    p["speed_ms"] = np.sqrt(dx ** 2 + dy ** 2) * fps / w
    sp = p.groupby("frame_idx").speed_ms.agg(["mean", "max"]).reindex(range(n_frames))
    out["mean_speed_ms"] = _smooth(sp["mean"], cfg["features"]["smooth_window_frames"])
    out["max_speed_ms"] = _smooth(sp["max"], cfg["features"]["smooth_window_frames"])
    p["sprinting"] = (p.speed_ms >= cfg["features"]["sprint_threshold_ms"]).astype(float)
    out["frac_above_sprint"] = p.groupby("frame_idx").sprinting.mean().reindex(range(n_frames))

    out = out.reset_index(names="frame_idx")
    out["timestamp_s"] = frame_to_time(out.frame_idx, fps)
    out["shot_id"] = shot_of
    for c in ["n_players_registered", "n_inside_65", "n_inside_45"]:
        out[c] = out[c].fillna(0).astype("int64")
    for c in out.columns:
        if out[c].dtype == "float64" and c != "timestamp_s":
            out[c] = out[c].astype("float32")
    return out


CANONICAL_COLS = ["frame_idx", "timestamp_s", "n_tracks", "max_bbox_height",
                  "mean_bbox_height", "cluster_density", "centroid_spread",
                  "mean_vertical_velocity", "max_vertical_velocity",
                  "scene_change_flag", "valid_frame", "feature_space"]


def canonical_features(img: pd.DataFrame, pit: pd.DataFrame | None) -> pd.DataFrame:
    """Join the two halves into the features table s07/s14/query.py read.

    cluster_density and centroid_spread are the two variables that exist in both
    spaces. Pitch space wins when it is available, because the paper's variables
    are defined in metres; pixels are the fallback, not the preference. The unit
    travels with the data in feature_space rather than being inferred from which
    stages happened to run.
    """
    out = img.copy()
    if pit is None:
        out["cluster_density"] = img["cluster_density_px"]
        out["centroid_spread"] = img["centroid_spread_px"]
        out["feature_space"] = "pixel"
    else:
        p = pit.set_index("frame_idx")
        out = out.set_index("frame_idx")
        out["cluster_density"] = p["cluster_density_m"]
        # The image analogue is sqrt(sx^2 + sy^2); mirror it in metres so the
        # two spaces mean the same thing and the rule thresholds stay comparable.
        out["centroid_spread"] = np.sqrt(
            p["spread_x_m"].astype(float) ** 2 + p["spread_y_m"].astype(float) ** 2
        ).astype("float32")
        out = out.reset_index()
        out["feature_space"] = "pitch"
    for c in ("cluster_density", "centroid_spread"):
        out[c] = out[c].astype("float32")
    return out[CANONICAL_COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--image-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s05_features", od / "logs")

    tr = read_table("tracks", od / "tracks.parquet")
    n_frames = int(tr.frame_idx.max()) + 1

    img = image_features(tr, cfg, n_frames)
    write_table(img, "features_image", od / "features_image.parquet")
    log.info("image features: %d frames | valid %.0f%% | cuts %d",
             len(img), 100 * img.valid_frame.mean(), int(img.scene_change_flag.sum()))

    if args.image_only:
        write_table(canonical_features(img, None), "features", od / "features.parquet")
        log.info("features.parquet written in PIXEL space (feature_space=pixel). "
                 "The paper's variables are defined in metres — this is enough to "
                 "detect candidate events, not to code them.")
        log.info("stopping before pitch space. Next: s04_register_pitch.py --list-shots")
        make_provenance("s05_features", cfg, ["tracks.parquet"],
                        {**cfg["features"], "feature_space": "pixel",
                         "image_only": True}).write(od / "provenance_s05.json")
        return

    hpath = od / "homographies.json"
    if not hpath.exists():
        raise SystemExit("no homographies.json — run s04_register_pitch.py --fit first, "
                         "or pass --image-only to work in pixels only")
    Hs = json.loads(hpath.read_text())
    shots = pd.read_csv(od / "shots.csv")
    pit = pitch_features(tr, cfg, shots, Hs, n_frames, log)
    write_table(pit, "features_pitch", od / "features_pitch.parquet")
    write_table(canonical_features(img, pit), "features", od / "features.parquet")

    log.info("pitch features written | median players inside the 65 = %.1f",
             float(np.nanmedian(pit.n_inside_65)))
    log.info("SANITY: outside a press this should be low; during an opposition kickout "
             "the paper reports teams committing 8-11+ players inside the 65.")
    make_provenance("s05_features", cfg, ["tracks.parquet"],
                    {**cfg["features"], "feature_space": "pitch",
                     "image_only": False}).write(od / "provenance_s05.json")


if __name__ == "__main__":
    main()
