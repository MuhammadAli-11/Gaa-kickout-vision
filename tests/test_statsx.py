"""Tests for the statistics module.

These are the tests that matter: they check the numbers that go in the
report against values with a known answer, and cross-check the
hand-rolled ICC against pingouin. If a supervisor asks "how do you know
your ICC is right", this file is the answer.
"""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.statsx import (icc_2_1, bland_altman, cohens_kappa, wilson_ci, prf,
                        average_precision, bootstrap_ci)
from lib.matching import temporal_iou, match_events, events_to_bins
import pandas as pd


def test_wilson_bounds():
    lo, hi = wilson_ci(15, 15)
    assert hi <= 1.0 and lo < 1.0          # never exceeds 1, unlike Wald
    lo, hi = wilson_ci(0, 12)
    assert lo >= 0.0


def test_prf_known():
    r = prf(tp=8, fp=2, fn=4)
    assert r["precision"].value == pytest.approx(0.8)
    assert r["recall"].value == pytest.approx(8 / 12)
    assert r["f1"].value == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_icc_perfect_agreement():
    x = np.arange(20, dtype=float)
    icc = icc_2_1(np.stack([x, x], 1))
    assert icc.value == pytest.approx(1.0, abs=1e-6)


def test_icc_systematic_offset_penalised():
    """Absolute-agreement ICC must punish a constant bias.

    This is the whole reason ICC(2,1) is used instead of Pearson r:
    a model that is reliably 2 s long is perfectly CORRELATED with the
    human and not interchangeable with them.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1.5, 40)
    icc_biased = icc_2_1(np.stack([x, x + 2.0], 1)).value
    icc_clean = icc_2_1(np.stack([x, x], 1)).value
    assert icc_biased < icc_clean
    assert np.corrcoef(x, x + 2.0)[0, 1] == pytest.approx(1.0)


@pytest.mark.skipif("pingouin" not in [m for m in __import__("sys").modules] and
                    not __import__("importlib").util.find_spec("pingouin"),
                    reason="pingouin not installed")
def test_icc_matches_pingouin():
    import pandas as pd
    import pingouin as pg
    rng = np.random.default_rng(7)
    a = rng.normal(4, 1.2, 25)
    b = a + rng.normal(0.3, 0.6, 25)
    mine = icc_2_1(np.stack([a, b], 1)).value
    long = pd.DataFrame({"targets": list(range(25)) * 2, "raters": ["a"] * 25 + ["b"] * 25,
                         "scores": np.concatenate([a, b])})
    tbl = pg.intraclass_corr(data=long, targets="targets", raters="raters",
                             ratings="scores").set_index("Type")
    theirs = tbl.loc["ICC(A,1)", "ICC"]        # ICC(2,1) in Shrout & Fleiss notation
    assert mine == pytest.approx(theirs, abs=1e-6)
    ci = icc_2_1(np.stack([a, b], 1))
    assert [round(ci.ci_low, 2), round(ci.ci_high, 2)] == list(tbl.loc["ICC(A,1)", "CI95"])


def test_bland_altman_recovers_bias():
    rng = np.random.default_rng(3)
    human = rng.normal(4.0, 0.8, 60)
    model = human + 0.5 + rng.normal(0, 0.3, 60)
    ba = bland_altman(model, human)
    assert ba.bias == pytest.approx(0.5, abs=0.12)
    assert ba.loa_lower < ba.bias < ba.loa_upper
    assert ba.bias_ci[0] < 0.5 < ba.bias_ci[1]


def test_kappa_known_table():
    y1 = np.array([1] * 20 + [0] * 80)
    y2 = np.array([1] * 15 + [0] * 5 + [1] * 5 + [0] * 75)
    k = cohens_kappa(y1, y2)
    assert 0.5 < k.value < 0.85
    assert k.ci_low < k.value < k.ci_high


def test_kappa_chance_is_zero():
    rng = np.random.default_rng(11)
    y1 = rng.integers(0, 2, 5000)
    y2 = rng.integers(0, 2, 5000)
    assert abs(cohens_kappa(y1, y2).value) < 0.05


def test_temporal_iou():
    m = temporal_iou([[0, 10]], [[5, 15]])
    assert m[0, 0] == pytest.approx(5 / 15)
    assert temporal_iou([[0, 1]], [[8, 9]])[0, 0] == 0.0


def test_match_events_one_to_one():
    """Two predictions on one GT event must yield 1 TP and 1 FP, not 2 TPs."""
    pred = pd.DataFrame(dict(event_id=["p1", "p2"], t_start_s=[10.0, 10.2], t_end_s=[14.0, 14.1],
                             t_peak_s=[12.0, 12.1], confidence=[0.9, 0.8], n_players_in_contest=[4, 4]))
    gt = pd.DataFrame(dict(event_id=["g1"], t_start_s=[10.0], t_end_s=[14.0], t_peak_s=[12.0],
                           n_players_in_contest=[4]))
    m = match_events(pred, gt, 0.5)
    assert (m.kind == "TP").sum() == 1
    assert (m.kind == "FP").sum() == 1


def test_average_precision_perfect():
    ap, _, _ = average_precision([0.9, 0.8, 0.7], [1, 1, 1], n_gt=3)
    assert ap == pytest.approx(1.0)


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(5)
    data = rng.normal(3.0, 1.0, 200)
    pt, lo, hi = bootstrap_ci(data, lambda a: float(np.mean(a.astype(float))), n_resamples=2000)
    assert lo < 3.0 < hi
    assert pt == pytest.approx(data.mean(), abs=1e-9)


def test_events_to_bins():
    df = pd.DataFrame(dict(t_start_s=[0.0, 20.0], t_end_s=[4.0, 26.0]))
    v = events_to_bins(df, total_s=30, bin_s=5)
    assert v.tolist() == [1, 0, 0, 0, 1, 1]


# ---------------------------------------------------------------------
# Weighted kappa and pitch geometry — added when the project anchored on
# McColgan et al. (2026), whose defender bands are ORDERED and whose
# variables are all defined in pitch metres.
# ---------------------------------------------------------------------
from lib.statsx import weighted_kappa                       # noqa: E402
from lib.pitch import Pitch, fit_homography, apply_homography, reprojection_error_m, foot_point  # noqa: E402

BANDS = ["0-4", "5-7", "8-10", "11+"]


def test_weighted_kappa_perfect():
    y = ["0-4", "5-7", "8-10", "11+"] * 5
    assert weighted_kappa(y, y, BANDS).value == pytest.approx(1.0)


def test_weighted_kappa_punishes_distant_confusions_more():
    """The whole reason to weight: confusing 0-4 with 11+ is worse than
    confusing 8-10 with 11+, because the bands are an ordered dose."""
    truth = ["8-10"] * 20
    near = ["11+"] * 5 + ["8-10"] * 15
    far = ["0-4"] * 5 + ["8-10"] * 15
    # with a single true category kappa is degenerate, so use a spread truth
    truth = (["0-4"] * 5 + ["5-7"] * 5 + ["8-10"] * 5 + ["11+"] * 5)
    near = (["5-7"] * 5 + ["5-7"] * 5 + ["8-10"] * 5 + ["11+"] * 5)     # off by one band
    far = (["11+"] * 5 + ["5-7"] * 5 + ["8-10"] * 5 + ["11+"] * 5)      # off by three
    assert weighted_kappa(truth, near, BANDS).value > weighted_kappa(truth, far, BANDS).value


def test_weighted_kappa_beats_unweighted_on_near_misses():
    truth = ["0-4"] * 5 + ["5-7"] * 5 + ["8-10"] * 5 + ["11+"] * 5
    pred = ["5-7"] * 5 + ["8-10"] * 5 + ["11+"] * 5 + ["8-10"] * 5      # all off by exactly one
    wk = weighted_kappa(truth, pred, BANDS).value
    unweighted = cohens_kappa((np.array(truth) == "0-4").astype(int),
                              (np.array(pred) == "0-4").astype(int)).value
    assert wk > unweighted


def test_pitch_line_positions():
    p = Pitch(length=145.0, width=88.0)
    assert p.line_x(65.0, "attacking") == pytest.approx(145 / 2 - 65)
    assert p.line_x(65.0, "defending") == pytest.approx(-145 / 2 + 65)
    assert p.inside_65(20.0, "attacking") is True
    assert p.inside_65(-20.0, "attacking") is False


def test_homography_roundtrip_is_exact_without_noise():
    """A synthetic camera, inverted, must return the original metres."""
    p = Pitch()
    L, W = p.length / 2, p.width / 2
    src = np.array([[-L, -W], [L, -W], [L, W], [-L, W]], float)
    dst = np.array([[60, 640], [1220, 640], [980, 300], [300, 300]], float)
    H_p2i, _ = fit_homography(src, dst)
    H_i2p = np.linalg.inv(H_p2i)
    back = apply_homography(H_i2p, dst)
    assert np.allclose(back, src, atol=1e-6)


def test_reprojection_error_is_reported_in_metres():
    p = Pitch()
    L, W = p.length / 2, p.width / 2
    src = np.array([[-L, -W], [L, -W], [L, W], [-L, W], [0, 0]], float)
    dst = np.array([[60, 640], [1220, 640], [980, 300], [300, 300], [640, 430]], float)
    H, _ = fit_homography(dst, src)
    e = reprojection_error_m(H, dst, src)
    assert e["n"] == 5 and e["rmse_m"] < 25.0     # sane magnitude, in metres not pixels


def test_foot_point_is_bottom_centre_not_centroid():
    """Homographies map the ground plane; a player's centroid floats above it."""
    fx, fy = foot_point(100.0, 200.0, 140.0, 300.0)
    assert (fx, fy) == (120.0, 300.0)
