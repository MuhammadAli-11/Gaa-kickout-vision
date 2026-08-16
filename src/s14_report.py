#!/usr/bin/env python3
"""S11 — Assemble every metric into figures, markdown tables and one metrics.json.

Nothing in report/validation_report.md should be typed by hand. Numbers
are pasted from generated tables, so a re-run with different thresholds
cannot leave a stale figure in the write-up. That property is worth
mentioning out loud in the interview.

Figures produced:
  fig1_pr_events.png        PR curve for event detection, AP annotated
  fig2_bland_altman.png     duration and t_peak agreement, bias + LoA
  fig3_recall_by_size.png   detection recall by bbox-area tercile, Wilson CIs
  fig4_feature_timeline.png feature series with GT and predicted events marked
  fig5_failure_causes.png   error type x cause cross-tab

Usage:
    python src/s11_report.py --video-id gaa_lf_r3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.config import load_config, out_dir, resolve
from lib.logging_setup import get_logger
from lib.schema import read_table

plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def _load(od: Path, name: str) -> dict:
    p = od / name
    return json.loads(p.read_text()) if p.exists() else {}


def fig_pr(od: Path, ev: dict) -> None:
    p = od / "pr_curve_events.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.step(d.recall, d.precision, where="post", lw=2)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Kickout-contest detection (tIoU 0.5)\nAP = {ev.get('average_precision', float('nan')):.3f}, "
                 f"n = {ev.get('n_gt_events', 0)} events")
    fig.tight_layout(); fig.savefig(od / "figures" / "fig1_pr_events.png"); plt.close(fig)


def fig_bland_altman(od: Path, ag: dict) -> None:
    mv = ag.get("model_vs_human", {})
    m = od / "event_matches.csv"
    if "bland_altman_duration_s" not in mv or not m.exists():
        return
    d = pd.read_csv(m).query("kind == 'TP'")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, (a, b, key, title) in zip(axes, [
        (d.dur_pred, d.dur_gt, "bland_altman_duration_s", "Contest duration (s)"),
        (d.t_peak_pred, d.t_peak_gt, "bland_altman_t_peak_s", "Contest peak time (s)")]):
        ba = mv[key]
        diff, mean = a - b, (a + b) / 2
        ax.scatter(mean, diff, s=26, alpha=0.85, zorder=3)
        ax.axhline(ba["bias"], ls="-", lw=1.2, label=f"bias {ba['bias']:+.2f}")
        ax.axhline(ba["loa_upper"], ls="--", lw=1, label=f"95% LoA [{ba['loa_lower']:+.2f}, {ba['loa_upper']:+.2f}]")
        ax.axhline(ba["loa_lower"], ls="--", lw=1)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(f"{title}  (n={ba['n']})"); ax.set_xlabel("Mean of model and human")
        ax.set_ylabel("Model − human"); ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(od / "figures" / "fig2_bland_altman.png"); plt.close(fig)


def fig_recall_by_size(od: Path, det: dict) -> None:
    r = det.get("recall_by_size")
    if not r:
        return
    ks = ["small", "medium", "large"]
    vals = [r[k]["recall"] for k in ks]
    err = np.array([[v - r[k]["ci"][0] for v, k in zip(vals, ks)],
                    [r[k]["ci"][1] - v for v, k in zip(vals, ks)]])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.bar(ks, vals, yerr=err, capsize=4, width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, 0.02, f"n={r[k]['n']}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Recall @ IoU 0.5")
    ax.set_title("Detection recall by player bbox-area tercile\n(95% Wilson intervals)")
    fig.tight_layout(); fig.savefig(od / "figures" / "fig3_recall_by_size.png"); plt.close(fig)


def fig_timeline(od: Path, gt: pd.DataFrame) -> None:
    f = read_table("features", od / "features.parquet")
    pred = read_table("events_pred", od / "events_pred.csv")
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.4), sharex=True)
    axes[0].plot(f.timestamp_s, f.centroid_spread, lw=0.8); axes[0].set_ylabel("centroid\nspread (px)")
    axes[1].plot(f.timestamp_s, f.mean_vertical_velocity, lw=0.8); axes[1].set_ylabel("vertical\nvel (px/s)")
    axes[2].plot(f.timestamp_s, f.n_tracks, lw=0.8); axes[2].set_ylabel("n tracks"); axes[2].set_xlabel("time (s)")
    for ax in axes:
        for _, r in gt.iterrows():
            ax.axvspan(r.t_start_s, r.t_end_s, color="tab:green", alpha=0.18, lw=0)
        for _, r in pred.iterrows():
            ax.axvline(r.t_peak_s, color="tab:red", lw=0.8, alpha=0.7)
    axes[0].set_title("Scene features with manual coding (green) and predictions (red)")
    fig.tight_layout(); fig.savefig(od / "figures" / "fig4_feature_timeline.png"); plt.close(fig)


def fig_failures(od: Path) -> pd.DataFrame | None:
    p = od / "failure_audit.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.cause.notna() & (d.cause.astype(str).str.len() > 0)]
    if d.empty:
        return None
    ct = pd.crosstab(d.cause, d.error_type)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ct.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("cases"); ax.set_ylabel(""); ax.set_title(f"Failure causes (n={len(d)})")
    fig.tight_layout(); fig.savefig(od / "figures" / "fig5_failure_causes.png"); plt.close(fig)
    return ct


def md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(e: dict, dp: int = 3) -> str:
    return f"{e['value']:.{dp}f} [{e['ci_low']:.{dp}f}, {e['ci_high']:.{dp}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    (od / "figures").mkdir(exist_ok=True)
    log = get_logger("s11_report", od / "logs")

    det, ev, ag = _load(od, "metrics_detection.json"), _load(od, "metrics_events.json"), _load(od, "metrics_agreement.json")
    gt = pd.read_csv(resolve(cfg, "gt") / args.video_id / "gt_events.csv").query("coding_pass == 1")

    fig_pr(od, ev); fig_bland_altman(od, ag); fig_recall_by_size(od, det)
    fig_timeline(od, gt); ct = fig_failures(od)

    # ---- markdown tables ------------------------------------------
    tables = {}
    if det:
        tables["t1_detection"] = md_table([
            ["Precision @ IoU 0.5", fmt(det["precision"])],
            ["Recall @ IoU 0.5", fmt(det["recall"])],
            ["F1 @ IoU 0.5", fmt(det["f1"])],
            ["mAP@0.5", f"{det.get('mAP@0.5', float('nan')):.3f}"],
            ["Frames annotated", det.get("n_frames_annotated", "-")],
            ["GT boxes", det.get("n_gt_boxes", "-")],
        ], ["Detection metric", "Value [95% CI]"])
        tables["t2_size"] = md_table(
            [[s, det["recall_by_size"][s]["n"],
              f"{det['recall_by_size'][s]['recall']:.3f} "
              f"[{det['recall_by_size'][s]['ci'][0]:.3f}, {det['recall_by_size'][s]['ci'][1]:.3f}]"]
             for s in ["small", "medium", "large"]],
            ["bbox-area tercile", "n GT boxes", "Recall [95% CI]"])
        if isinstance(det.get("tracking"), dict) and "mota" in det["tracking"]:
            t = det["tracking"]
            tables["t3_tracking"] = md_table(
                [[k.upper(), f"{t[k]:.3f}" if isinstance(t[k], float) else t[k]]
                 for k in ["mota", "idf1", "num_switches", "num_fragmentations",
                           "mostly_tracked", "mostly_lost"] if k in t],
                ["Tracking metric", "Value"])
    if ev:
        tables["t4_events"] = md_table(
            [[k, v["tp"], v["fp"], v["fn"], fmt(v["precision"]), fmt(v["recall"]), fmt(v["f1"])]
             for k, v in ev["by_tiou"].items()],
            ["tIoU", "TP", "FP", "FN", "Precision [95% CI]", "Recall [95% CI]", "F1 [95% CI]"])
        if "temporal_offset_s" in ev:
            o = ev["temporal_offset_s"]
            tables["t5_offset"] = md_table(
                [["Mean |offset|", f"{o['mean_abs']:.2f} s"], ["Median |offset|", f"{o['median_abs']:.2f} s"],
                 ["Signed mean", f"{o['signed_mean']:+.2f} s"], ["95th pct |offset|", f"{o['p95_abs']:.2f} s"],
                 ["n matched events", o["n"]]],
                ["Temporal localisation", "Value"])
    if ag and "model_vs_human" in ag and "icc_duration" in ag["model_vs_human"]:
        mv = ag["model_vs_human"]
        rows = [["ICC(2,1), contest duration", fmt(mv["icc_duration"])],
                ["ICC(2,1), n players in contest", fmt(mv["icc_n_players"])],
                ["Cohen's kappa, presence (5 s bins)", fmt(ag["kappa_model_vs_human"])]]
        ceil = ag.get("intra_rater_ceiling", {})
        if "kappa" in ceil:
            rows += [["**Intra-rater kappa (ceiling)**", f"**{fmt(ceil['kappa'])}**"],
                     ["Model kappa as % of ceiling",
                      f"{100 * ag['model_relative_to_ceiling']['kappa_ratio']:.0f}%"]]
        tables["t6_agreement"] = md_table(rows, ["Agreement metric", "Value [95% CI]"])
    if ct is not None:
        tables["t7_failures"] = ct.to_markdown()

    tdir = od / "tables"; tdir.mkdir(exist_ok=True)
    for name, tbl in tables.items():
        (tdir / f"{name}.md").write_text(tbl + "\n")

    all_metrics = {"detection": det, "events": ev, "agreement": ag,
                   "run_id": od.name, "config": {k: cfg[k] for k in ["video", "detect", "track", "features", "events", "eval"]}}
    (od / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=float))
    log.info("wrote %d tables and %d figures", len(tables), len(list((od / 'figures').glob('*.png'))))
    log.info("paste tables from %s into report/validation_report.md", tdir)


if __name__ == "__main__":
    main()
