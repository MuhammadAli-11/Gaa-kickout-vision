#!/usr/bin/env python3
"""S11 — Agreement with the paper's coding scheme, variable by variable.

This is the headline analysis. For each variable in the McColgan et al.
(2026) scheme that the pipeline attempts, it reports agreement between
automated and manual coding using the statistic appropriate to that
variable's measurement level — and against two reference points, not one:

  1. YOUR OWN intra-rater agreement (blind re-code, 48 h apart). The
     ceiling for this project's labels.
  2. THE PUBLISHED inter-operator agreement. The source paper reports
     Cohen's kappa between 0.981 and 1.00 across kickout events, outcome
     and strategies, on 200 kickouts against a Level 3 GAA analyst.

Read point 2 carefully before quoting it, because the comparison is not
like for like and an examiner will know it. Their coders agreed on the
CLASSIFICATION of kickouts already located and tagged by a human in
NacSport. This pipeline has to locate the kickout in time, register the
pitch, detect players, assign them to teams, and only then classify. Each
of those stages contributes error the published figure never had to
absorb. Their kappa is the right benchmark for the classification step
alone, and the wrong one for the end-to-end system. State which you mean.

Statistic by measurement level:
  count    n_defenders_in_65   -> ICC(2,1), Bland-Altman, exact-match rate
  ordinal  defender band       -> linear weighted kappa, exact and +/-1
  nominal  defensive strategy  -> kappa, per-class recall, confusion matrix
  binary   short/long, contested -> kappa, precision, recall

Usage:
    python src/s11_evaluate_coding.py --video-id <id> [--source gt_timestamps]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir
from lib.logging_setup import get_logger
from lib.schema import read_table
from lib.statsx import (icc_2_1, bland_altman, cohens_kappa, weighted_kappa,
                        prf, wilson_ci, ceiling_adjusted)

PUBLISHED_KAPPA_RANGE = (0.981, 1.00)
PUBLISHED_EXCLUDED, PUBLISHED_TOTAL = 908, 3081
PUBLISHED_CITATION = (
    "McColgan, A.; Bradley, J.; Earle, D.; Gaul, D.; Martin, D. (2026). Identifying and "
    "Defining Kickout Strategies in Senior Inter-County Ladies Gaelic Football. "
    "Applied Sciences 16(7), 3277. https://doi.org/10.3390/app16073277")


def nominal_agreement(pred: pd.Series, gt: pd.Series, alpha: float) -> dict:
    labels = sorted(set(pred.dropna().astype(str)) | set(gt.dropna().astype(str)))
    cm = pd.crosstab(gt.astype(str), pred.astype(str)).reindex(
        index=labels, columns=labels, fill_value=0)
    n = int(cm.values.sum())
    exact = int(np.trace(cm.values))
    po = exact / n if n else np.nan
    pe = float((cm.sum(axis=0).values * cm.sum(axis=1).values).sum()) / (n ** 2) if n else np.nan
    kappa = (po - pe) / (1 - pe) if (n and pe != 1) else np.nan
    per_class = {c: {"n": int(cm.loc[c].sum()),
                     "recall": float(cm.loc[c, c] / cm.loc[c].sum()) if cm.loc[c].sum() else np.nan}
                 for c in labels}
    lo, hi = wilson_ci(exact, n, alpha)
    return {"n": n, "exact_agreement": po, "exact_ci": [lo, hi], "kappa": kappa,
            "p_expected": pe, "confusion_matrix": cm.to_dict(), "per_class": per_class}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--source", default="gt_timestamps",
                    choices=["gt_timestamps", "predicted_timestamps"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s11_coding", od / "logs")
    E = cfg["eval"]

    pred = pd.read_csv(od / f"kickout_coding_{args.source}.csv")
    gt_all = pd.read_csv(resolve(cfg, "gt") / args.video_id / "gt_coding.csv")
    gt1, gt2 = gt_all[gt_all.coding_pass == 1], gt_all[gt_all.coding_pass == 2]

    m = pred.merge(gt1, on="event_id", suffixes=("_pred", "_gt"))
    m = m[m.codeable == 1]
    log.info("%d kickouts with both automated and manual coding", len(m))
    if len(m) < 8:
        log.warning("fewer than 8 paired kickouts — every interval below will be wider than "
                    "the range of plausible values. Report n before any coefficient.")

    out: dict = {
        "n_paired": int(len(m)), "coding_source": args.source,
        "published_reference": {
            "kappa_range": list(PUBLISHED_KAPPA_RANGE), "citation": PUBLISHED_CITATION,
            "caveat": ("Published value is inter-operator agreement on classification of "
                       "human-located kickouts, not end-to-end automated coding.")}}

    # ---------- PRIMARY: defender count inside the 65 ---------------
    if {"n_defenders_in_65_pred", "n_defenders_in_65_gt"} <= set(m.columns) and len(m) >= 3:
        a = m.n_defenders_in_65_pred.astype(float).values
        b = m.n_defenders_in_65_gt.astype(float).values
        icc = icc_2_1(np.stack([a, b], 1), E["alpha"])
        ba = bland_altman(a, b, E["alpha"])
        exact, within1 = int((a == b).sum()), int((np.abs(a - b) <= 1).sum())
        out["n_defenders_in_65"] = {
            "icc": icc.to_dict(), "bland_altman": ba.to_dict(),
            "exact_match": {"value": exact / len(m),
                            "ci": list(wilson_ci(exact, len(m), E["alpha"])), "n": len(m)},
            "within_1": {"value": within1 / len(m),
                         "ci": list(wilson_ci(within1, len(m), E["alpha"])), "n": len(m)},
            "mean_abs_error": float(np.abs(a - b).mean()),
            "median_unassigned_uncertainty": float(m.count_uncertainty.median()),
        }
        log.info("PRIMARY defenders inside the 65 | %s", icc)
        log.info("  bias %+.2f players, 95%% LoA [%+.2f, %+.2f], exact %.0f%%, within 1 %.0f%%",
                 ba.bias, ba.loa_lower, ba.loa_upper,
                 100 * exact / len(m), 100 * within1 / len(m))
        log.info("  INTERPRET AGAINST THEIR BANDS: the paper's model contrasts 0-7, 8-10 and "
                 "11+ defenders. An error of +/-1 rarely crosses a band; +/-3 routinely does "
                 "and would change the modelled effect on kickout outcome.")

    # ---------- ordinal: defender band ------------------------------
    if {"n_defenders_band_pred", "n_defenders_band_gt"} <= set(m.columns) and len(m) >= 3:
        order = [b[2] for b in cfg["coding"]["defender_bands"]]
        wk = weighted_kappa(m.n_defenders_band_gt, m.n_defenders_band_pred, order, weights="linear")
        out["n_defenders_band"] = {
            "weighted_kappa": wk.to_dict(),
            **nominal_agreement(m.n_defenders_band_pred, m.n_defenders_band_gt, E["alpha"])}
        log.info("defender band | %s", wk)

    # ---------- nominal: defensive strategy -------------------------
    if {"defensive_strategy_pred", "defensive_strategy_gt"} <= set(m.columns) and len(m) >= 3:
        res = nominal_agreement(m.defensive_strategy_pred, m.defensive_strategy_gt, E["alpha"])
        out["defensive_strategy"] = res
        log.info("defensive strategy | exact %.0f%% [%.0f, %.0f], kappa %.3f (n=%d)",
                 100 * res["exact_agreement"], 100 * res["exact_ci"][0],
                 100 * res["exact_ci"][1], res["kappa"], res["n"])
        for c, v in res["per_class"].items():
            log.info("    %-18s n=%2d recall=%.2f", c, v["n"], v["recall"])
        log.info("  The zonal / player-to-player boundary is one threshold on median "
                 "nearest-opponent distance. If most errors sit between those two classes, "
                 "report the threshold sensitivity curve rather than a single accuracy.")

    # ---------- binary variables ------------------------------------
    for var, pos in [("distance_class", "short"), ("contested_proxy", "contested")]:
        if not {f"{var}_pred", f"{var}_gt"} <= set(m.columns) or len(m) < 3:
            continue
        yp = (m[f"{var}_pred"] == pos).astype(int).values
        yg = (m[f"{var}_gt"] == pos).astype(int).values
        k = cohens_kappa(yp, yg, E["alpha"])
        tp = int(((yp == 1) & (yg == 1)).sum())
        fp = int(((yp == 1) & (yg == 0)).sum())
        fn = int(((yp == 0) & (yg == 1)).sum())
        out[var] = {"kappa": k.to_dict(),
                    **{kk: v.to_dict() for kk, v in prf(tp, fp, fn, E["alpha"]).items()}}
        log.info("%s | %s", var, k)

    # ---------- the ceiling -----------------------------------------
    if len(gt2) >= 3:
        c = gt1.merge(gt2, on="event_id", suffixes=("_p1", "_p2"))
        ceil: dict = {"n_recoded": int(len(c))}
        if len(c) >= 3:
            ceil["icc_n_defenders"] = icc_2_1(np.stack([
                c.n_defenders_in_65_p1.astype(float).values,
                c.n_defenders_in_65_p2.astype(float).values], 1), E["alpha"]).to_dict()
            ceil["strategy_exact"] = float((c.defensive_strategy_p1 == c.defensive_strategy_p2).mean())
        out["intra_rater_ceiling"] = ceil
        log.info("INTRA-RATER CEILING | defender-count ICC=%.3f, strategy exact %.0f%% (n=%d)",
                 ceil.get("icc_n_defenders", {}).get("value", np.nan),
                 100 * ceil.get("strategy_exact", np.nan), len(c))
        if "n_defenders_in_65" in out:
            out["relative_to_ceiling"] = {"icc_ratio": ceiling_adjusted(
                out["n_defenders_in_65"]["icc"]["value"],
                ceil.get("icc_n_defenders", {}).get("value", np.nan))}
    else:
        log.warning("no blind re-code found — without the ceiling the model numbers have no "
                    "scale. See docs/03 section C.")

    # ---------- coverage: the direct comparison with the paper -------
    total, codeable = len(pred), int((pred.codeable == 1).sum())
    out["coverage"] = {
        "n_kickouts_attempted": total, "n_codeable": codeable,
        "exclusion_rate": 1 - codeable / total if total else np.nan,
        "published_exclusion_rate": PUBLISHED_EXCLUDED / PUBLISHED_TOTAL,
        "note": ("The source paper excluded 908 of 3081 kickouts (29.4%) because broadcast "
                 "camera angles and score replays hid them. This pipeline's exclusion rate "
                 "measures the same limitation automatically.")}
    log.info("COVERAGE | pipeline coded %.0f%% of kickouts; the paper's manual coding used "
             "%.0f%% of theirs", 100 * codeable / max(total, 1),
             100 * (1 - PUBLISHED_EXCLUDED / PUBLISHED_TOTAL))

    (od / "metrics_coding.json").write_text(json.dumps(out, indent=2, default=float))
    log.info("wrote metrics_coding.json")


if __name__ == "__main__":
    main()
