"""Statistical analysis primitives.

Implemented here rather than called blind from a library so that every
number in the report can be explained at a whiteboard. `pingouin` is
still in requirements.txt and tests/test_statsx.py cross-checks the ICC
against it — agreement between an independent implementation and a
published package is the cheapest possible correctness evidence.

Everything returns a point estimate AND an interval. With n ~ 15 events
a point estimate on its own is close to meaningless, and reporting it
alone is the fastest way to lose credibility (see docs/07).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------
@dataclass
class Estimate:
    """A point estimate with an interval and the n it rests on."""
    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.name} = {self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] (n={self.n})"


# ---------------------------------------------------------------------
# Proportions — precision, recall, F1 with intervals
# ---------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval. Correct near 0 and 1, where Wald is not.

    With n=15 events, precision of 12/15 has a Wald interval that runs
    past 1.0. Use Wilson and say why.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def prf(tp: int, fp: int, fn: int, alpha: float = 0.05) -> dict[str, Estimate]:
    """Precision, recall, F1. Intervals: Wilson for P and R, bootstrap for F1."""
    p_n, r_n = tp + fp, tp + fn
    p = tp / p_n if p_n else float("nan")
    r = tp / r_n if r_n else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    p_lo, p_hi = wilson_ci(tp, p_n, alpha)
    r_lo, r_hi = wilson_ci(tp, r_n, alpha)
    # F1 interval from the corners of the P/R box: deliberately conservative.
    f_corners = [2 * a * b / (a + b) if (a + b) else 0.0
                 for a in (p_lo, p_hi) for b in (r_lo, r_hi)]
    return {
        "precision": Estimate("precision", p, p_lo, p_hi, p_n, "Wilson score"),
        "recall": Estimate("recall", r, r_lo, r_hi, r_n, "Wilson score"),
        "f1": Estimate("f1", f1, min(f_corners), max(f_corners), tp + fp + fn,
                       "propagated from Wilson P/R corners"),
    }


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------
def bootstrap_ci(
    data: Sequence,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 1729,
) -> tuple[float, float, float]:
    """Percentile bootstrap over the resampling unit given.

    IMPORTANT: the unit must be the EVENT, not the frame. Frames inside
    one kickout are not independent; resampling them fakes precision.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=object)
    n = len(arr)
    if n == 0:
        return (float("nan"),) * 3
    point = statistic(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = np.array([statistic(arr[i]) for i in idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------
# Agreement — ICC
# ---------------------------------------------------------------------
def icc_2_1(ratings: np.ndarray, alpha: float = 0.05) -> Estimate:
    """ICC(2,1): two-way random effects, absolute agreement, single measure.

    ratings: (n_targets, k_raters). Rows = kickouts, columns = coders
    (here: your manual coding vs the model, or pass 1 vs pass 2).

    Model choice matters and gets asked about. ICC(2,1) is right here
    because (a) both 'raters' rate every event, (b) we care about
    ABSOLUTE agreement not just rank consistency — a model that is
    systematically 0.4 s early is not interchangeable with a human even
    if correlation is perfect, and (c) we report the single-measure form
    because a single automated pass is what would be deployed.
    """
    Y = np.asarray(ratings, dtype=float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    if n < 2:
        return Estimate("ICC(2,1)", float("nan"), float("nan"), float("nan"), n, "two-way random, absolute")

    grand = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # between targets
    ms_c = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # between raters
    ss_t = ((Y - grand) ** 2).sum()
    ss_e = ss_t - ms_r * (n - 1) - ms_c * (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))

    if ms_e <= 0 or not np.isfinite(ms_e):      # perfect agreement: no residual variance
        return Estimate("ICC(2,1)", 1.0, 1.0, 1.0, n,
                        "two-way random effects, absolute agreement, single measure (exact)")

    icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)

    # Exact CI (McGraw & Wong 1996, Table 7, ICC(A,1))
    f_l = stats.f.ppf(1 - alpha / 2, n - 1, (n - 1) * (k - 1))
    f_u = stats.f.ppf(1 - alpha / 2, (n - 1) * (k - 1), n - 1)
    fj = ms_c / ms_e
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = ((k - 1) * (n - 1) * ((k * icc * fj + n * (1 + (k - 1) * icc) - k * icc)) ** 2) / (
            (n - 1) * k**2 * icc**2 * fj**2 + (n * (1 + (k - 1) * icc) - k * icc) ** 2
        )
        f_lo = stats.f.ppf(1 - alpha / 2, n - 1, nu)
        f_hi = stats.f.ppf(1 - alpha / 2, nu, n - 1)
        lo = n * (ms_r - f_lo * ms_e) / (
            f_lo * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r
        )
        hi = n * (f_hi * ms_r - ms_e) / (
            k * ms_c + (k * n - k - n) * ms_e + n * f_hi * ms_r
        )
    lo = float(np.clip(np.nan_to_num(lo, nan=-1.0), -1.0, 1.0))
    hi = float(np.clip(np.nan_to_num(hi, nan=1.0), -1.0, 1.0))
    return Estimate("ICC(2,1)", float(icc), lo, hi, n, "two-way random effects, absolute agreement, single measure")


# ---------------------------------------------------------------------
# Agreement — Bland-Altman
# ---------------------------------------------------------------------
@dataclass
class BlandAltman:
    bias: float
    bias_ci: tuple[float, float]
    sd_diff: float
    loa_lower: float
    loa_upper: float
    loa_lower_ci: tuple[float, float]
    loa_upper_ci: tuple[float, float]
    n: int
    prop_bias_slope: float
    prop_bias_p: float

    def to_dict(self) -> dict:
        return asdict(self)


def bland_altman(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> BlandAltman:
    """Bias and 95% limits of agreement for method A vs method B.

    Convention used throughout this repo: difference = model - human,
    so a POSITIVE bias means the model over-estimates. State the
    direction in every figure caption; reviewers cannot infer it.

    Also regresses difference on mean to test for proportional bias —
    if the model's error grows with contest duration, a single LoA is
    the wrong summary and you should say so rather than report it anyway.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n < 2:
        nanci = (float("nan"), float("nan"))
        return BlandAltman(float("nan"), nanci, float("nan"), float("nan"), float("nan"),
                           nanci, nanci, n, float("nan"), float("nan"))
    diff = a - b
    mean = (a + b) / 2
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    se_bias = sd / np.sqrt(n)
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    se_loa = np.sqrt((1 / n + 1.96**2 / (2 * (n - 1))) * sd**2)
    lr = stats.linregress(mean, diff)
    return BlandAltman(
        bias=bias,
        bias_ci=(bias - t * se_bias, bias + t * se_bias),
        sd_diff=sd,
        loa_lower=loa_lo,
        loa_upper=loa_hi,
        loa_lower_ci=(loa_lo - t * se_loa, loa_lo + t * se_loa),
        loa_upper_ci=(loa_hi - t * se_loa, loa_hi + t * se_loa),
        n=n,
        prop_bias_slope=float(lr.slope),
        prop_bias_p=float(lr.pvalue),
    )


# ---------------------------------------------------------------------
# Agreement — Cohen's kappa
# ---------------------------------------------------------------------
def cohens_kappa(y1: Sequence[int], y2: Sequence[int], alpha: float = 0.05) -> Estimate:
    """Cohen's kappa on binary presence vectors, with asymptotic CI.

    Note the base-rate caveat: kickout-present bins are a minority of
    all 5 s bins, so kappa is sensitive to bin width. Report the
    observed agreement, the expected agreement and the prevalence
    alongside kappa, never kappa alone (docs/07 §4.3).
    """
    y1 = np.asarray(y1, int); y2 = np.asarray(y2, int)
    n = len(y1)
    if n == 0:
        return Estimate("kappa", *[float("nan")] * 3, 0, "Cohen")
    a = int(((y1 == 1) & (y2 == 1)).sum())
    b = int(((y1 == 1) & (y2 == 0)).sum())
    c = int(((y1 == 0) & (y2 == 1)).sum())
    d = int(((y1 == 0) & (y2 == 0)).sum())
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n**2
    k = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if pe != 1 else float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return Estimate("kappa", float(k), float(k - z * se), float(k + z * se), n,
                    f"Cohen; po={po:.3f}, pe={pe:.3f}, prevalence={(a+b)/n:.3f}")


def weighted_kappa(y1, y2, categories: Sequence[str], weights: str = "linear",
                   alpha: float = 0.05) -> Estimate:
    """Weighted kappa for ORDERED categories.

    Used for the defender band (0-4 / 5-7 / 8-10 / 11+). Unweighted kappa
    treats confusing 0-4 with 11+ as no worse than confusing 8-10 with
    11+, which is plainly wrong when the categories are ordered counts —
    and the source paper's model treats those bands as an ordered dose,
    so the statistic should too.

    Linear weights penalise in proportion to the number of bands missed;
    quadratic penalise the square. Linear is the conservative default.
    """
    cats = list(categories)
    idx = {c: i for i, c in enumerate(cats)}
    a = np.array([idx.get(str(v), -1) for v in np.asarray(y1)])
    b = np.array([idx.get(str(v), -1) for v in np.asarray(y2)])
    keep = (a >= 0) & (b >= 0)
    a, b = a[keep], b[keep]
    n, k = len(a), len(cats)
    if n == 0 or k < 2:
        return Estimate("weighted kappa", *[float("nan")] * 3, n, weights)

    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    O /= n
    ra, rb = O.sum(axis=1), O.sum(axis=0)
    Ex = np.outer(ra, rb)

    i_, j_ = np.indices((k, k))
    d = np.abs(i_ - j_) / (k - 1)
    W = d if weights == "linear" else d ** 2

    num, den = (W * O).sum(), (W * Ex).sum()
    kappa = 1 - num / den if den else float("nan")

    # bootstrap interval: the closed form for weighted kappa is fragile
    # at the sample sizes involved here.
    rng = np.random.default_rng(1729)
    boots = []
    for _ in range(2000):
        s = rng.integers(0, n, n)
        Ob = np.zeros((k, k))
        for i, j in zip(a[s], b[s]):
            Ob[i, j] += 1
        Ob /= n
        Eb = np.outer(Ob.sum(axis=1), Ob.sum(axis=0))
        db = (W * Eb).sum()
        if db > 0:
            boots.append(1 - (W * Ob).sum() / db)
    lo, hi = (np.percentile(boots, [alpha / 2 * 100, (1 - alpha / 2) * 100])
              if boots else (float("nan"), float("nan")))
    return Estimate("weighted kappa", float(kappa), float(lo), float(hi), n,
                    f"{weights} weights, percentile bootstrap")


# ---------------------------------------------------------------------
# Average precision (for the event PR curve)
# ---------------------------------------------------------------------
def average_precision(scores: Sequence[float], labels: Sequence[int], n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """All-point interpolated AP, plus the PR curve arrays for plotting."""
    order = np.argsort(-np.asarray(scores, float))
    lab = np.asarray(labels, int)[order]
    tp = np.cumsum(lab == 1)
    fp = np.cumsum(lab == 0)
    recall = tp / n_gt if n_gt else np.zeros_like(tp, dtype=float)
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):     # monotone envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
    return ap, recall, precision


# ---------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------
def ceiling_adjusted(model_score: float, ceiling_score: float) -> float:
    """Model performance as a proportion of the human self-agreement ceiling.

    A model F1 of 0.62 against labels whose own intra-rater kappa is 0.71
    is not '0.62 of the way there'. Reporting the ratio makes the
    measurement limit explicit. Report BOTH numbers, never the ratio alone.
    """
    if not np.isfinite(ceiling_score) or ceiling_score == 0:
        return float("nan")
    return float(model_score / ceiling_score)
