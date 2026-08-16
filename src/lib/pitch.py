"""Pitch model, image->pitch homography, and zone assignment.

This module exists because McColgan et al. (2026) code every kickout
variable in PITCH space, not image space: players inside the opposition
65 m line, kickouts received inside or outside the 45 m line, an
opposition player within 2 m of the receiver, four players across the
21 m line. None of those is expressible in pixels. Replicating their
coding scheme therefore requires a homography, and the accuracy of that
homography is itself a result to report — it sets a floor on every
downstream variable.

Coordinate convention
---------------------
Pitch coordinates are metres, origin at the CENTRE of the pitch:
    x: -L/2 (defending end line) .. +L/2 (attacking end line)
    y: -W/2 (near touchline) .. +W/2 (far touchline)
Distances quoted in the paper ("the 65") are measured from the nearer
END LINE, so the 65 m line at the attacking end sits at x = L/2 - 65.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------
# Pitch model
# ---------------------------------------------------------------------
@dataclass
class Pitch:
    """Nominal pitch dimensions and cross-pitch line positions, in metres.

    GAA/LGFA pitches are a RANGE, not a fixed size (length 130-145 m,
    width 80-90 m), and line naming differs between codes: men's Gaelic
    football marks the 13/20/45/65, while LGFA usage in the source paper
    refers to the 21 m line. Set these from config for the venue and code
    you are actually analysing, and state the assumed dimensions in the
    report — a 145 m pitch modelled as 130 m puts every derived distance
    out by more than 10%.
    """
    length: float = 145.0
    width: float = 88.0
    line_distances: tuple[float, ...] = (13.0, 21.0, 45.0, 65.0)
    small_rect: tuple[float, float] = (14.0, 4.5)     # width, depth
    large_rect: tuple[float, float] = (19.0, 13.0)
    goal_width: float = 6.5

    # -- lines as x-coordinates ------------------------------------
    def line_x(self, distance_from_end: float, end: str = "attacking") -> float:
        half = self.length / 2
        return half - distance_from_end if end == "attacking" else -half + distance_from_end

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named points a human can unambiguously click in a broadcast frame.

        Only points that are actually VISIBLE and unambiguous are listed.
        Intersections of a cross-pitch line with a touchline are the most
        reliable; centre-circle tangents and rectangle corners help when
        the shot is tight.
        """
        L, W = self.length / 2, self.width / 2
        pts: dict[str, tuple[float, float]] = {
            "centre_spot": (0.0, 0.0),
            "halfway_near_touch": (0.0, -W),
            "halfway_far_touch": (0.0, W),
        }
        for end, sign in (("attacking", 1), ("defending", -1)):
            e = "att" if end == "attacking" else "def"
            pts[f"{e}_endline_near_corner"] = (sign * L, -W)
            pts[f"{e}_endline_far_corner"] = (sign * L, W)
            for d in self.line_distances:
                x = self.line_x(d, end)
                pts[f"{e}_{int(d)}m_near_touch"] = (x, -W)
                pts[f"{e}_{int(d)}m_far_touch"] = (x, W)
            sw, sd = self.small_rect
            pts[f"{e}_small_rect_near"] = (sign * (L - sd), -sw / 2)
            pts[f"{e}_small_rect_far"] = (sign * (L - sd), sw / 2)
            lw, ld = self.large_rect
            pts[f"{e}_large_rect_near"] = (sign * (L - ld), -lw / 2)
            pts[f"{e}_large_rect_far"] = (sign * (L - ld), lw / 2)
        return pts

    # -- the paper's zones -----------------------------------------
    def zone_of(self, x: float, y: float, end: str = "attacking") -> int | None:
        """Nine-zone grid in the attacking third, after Figure 1 of the paper.

        The paper divides the kickout landing area into nine zones (3
        across x 3 along) and reports that wide zones are more productive
        against a press. Reproduce the GRID from the published figure
        before using this — the exact boundaries are theirs, and a zone
        map that does not match theirs makes the comparison meaningless.

        Returns 1..9 (1 = near-touch/closest to the kicking end) or None
        if the point lies outside the zoned region.
        """
        x0 = self.line_x(65.0, end)
        x1 = self.line_x(0.0, end)
        lo, hi = min(x0, x1), max(x0, x1)
        if not (lo <= x <= hi):
            return None
        col = int(np.clip((x - lo) / ((hi - lo) / 3), 0, 2.999))
        row = int(np.clip((y + self.width / 2) / (self.width / 3), 0, 2.999))
        return int(row * 3 + col + 1)

    def inside_65(self, x: float, end: str = "attacking") -> bool:
        """The paper's key defensive count: players inside the opposition 65."""
        return x >= self.line_x(65.0, end) if end == "attacking" else x <= self.line_x(65.0, end)


# ---------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------
def fit_homography(image_pts: np.ndarray, pitch_pts: np.ndarray, ransac_px: float = 6.0):
    """Least-squares (or RANSAC) homography mapping image px -> pitch metres.

    Needs >= 4 correspondences; 6-8 well-spread points give a usable fit
    and leave some out for validation. Points clustered in one corner
    produce a homography that is excellent there and useless elsewhere,
    which is exactly the failure that inflates a reprojection error
    computed on the fitting points.
    """
    import cv2
    image_pts = np.asarray(image_pts, np.float32).reshape(-1, 1, 2)
    pitch_pts = np.asarray(pitch_pts, np.float32).reshape(-1, 1, 2)
    if len(image_pts) < 4:
        raise ValueError("need at least 4 point correspondences")
    method = 0 if len(image_pts) == 4 else cv2.RANSAC
    H, mask = cv2.findHomography(image_pts, pitch_pts, method, ransac_px)
    if H is None:
        raise ValueError("homography fit failed — check for collinear points")
    return H, (mask.ravel().astype(bool) if mask is not None else np.ones(len(image_pts), bool))


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map image points (N,2) to pitch metres (N,2)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty((0, 2))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = hom @ np.asarray(H, np.float64).T
    w = np.where(np.abs(out[:, 2:3]) < 1e-12, np.nan, out[:, 2:3])
    return out[:, :2] / w


def reprojection_error_m(H: np.ndarray, image_pts: np.ndarray, pitch_pts: np.ndarray) -> dict:
    """Error in METRES on the given correspondences.

    Report this on HELD-OUT landmarks, not on the ones used to fit. It is
    the honest uncertainty on every pitch-space variable downstream, and
    it is the number that decides whether a 65 m line call is meaningful:
    an RMSE of 3 m makes 'inside the 65' a coin-flip for anyone standing
    within 3 m of the line.
    """
    proj = apply_homography(H, image_pts)
    err = np.linalg.norm(proj - np.asarray(pitch_pts, float).reshape(-1, 2), axis=1)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return {"rmse_m": float("nan"), "median_m": float("nan"), "max_m": float("nan"), "n": 0}
    return {"rmse_m": float(np.sqrt((err ** 2).mean())), "median_m": float(np.median(err)),
            "max_m": float(err.max()), "n": int(len(err))}


def foot_point(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Ground-contact point of a player bounding box.

    Use the BOTTOM-CENTRE of the box, not the centroid. A homography maps
    the ground plane; the centroid of a standing player floats about a
    metre above it, and projecting it puts the player systematically
    further from the camera. For a leaping player even the foot point is
    wrong — which is a limitation to state, since leaping is precisely
    what happens at a contest.
    """
    return ((x1 + x2) / 2.0, y2)
