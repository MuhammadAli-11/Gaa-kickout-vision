#!/usr/bin/env python3
"""S08 — Code each kickout with the McColgan et al. (2026) variable set.

This is the stage the whole project is built around. For every kickout —
either the ones S07 located automatically, or your manually coded ones
when you want to isolate coding accuracy from detection accuracy — it
produces the variables the paper codes by hand in NacSport.

Which variables, and why those
------------------------------
PRIMARY. `n_defenders_in_65`, banded into the paper's categories.
    This is the right primary target for three reasons. It is the
    strongest predictor in their generalised linear mixed model — a zonal
    press with 11+ players inside the 65 was reported as roughly ten
    times more likely to win the opposition kickout than conceding. It is
    a pure counting-in-a-region task, so given a homography it needs no
    judgement. And because it needs no judgement, ground truth for it is
    reliable, which means a disagreement between model and human is
    informative about the model rather than about the labels.

SECONDARY. `defensive_strategy` in {zonal, player_to_player, concede}.
    Their definitions are geometric, which is what makes this tractable:
    player-to-player is defenders staying about a metre from an opponent,
    zonal is defenders occupying space instead, concede is not pressing
    at all. So the classifier is three rules on nearest-opponent distance
    and defender count — not a learned model, and defensible line by line
    against their published definitions.

SECONDARY. `distance_class` in {short, long}, split at the 45 m line
    where the ball is received.

EXPLORATORY. `contested`. The paper defines it as an opponent within 2 m
    of the player gaining possession. Without ball tracking there is no
    "player gaining possession", so what is computed here is a proxy:
    minimum opponent-to-opponent distance in the landing area at the
    contest peak. Report it as a proxy, and report the disagreement, do
    not quietly relabel the proxy with the paper's variable name.

NOT ATTEMPTED. Offensive strategy (Bunch and Break, Overload, Flat 4,
    Purposeful Movement, Limited Movement), kickout outcome, shots and
    scores. Offensive strategy depends on run intent and on a signal
    between goalkeeper and outfield players; outcome depends on the ball.
    Say so as a scope decision with a reason, rather than attempting them
    badly.

Usage:
    python src/s08_code_kickouts.py --video-id <id>                # code predicted events
    python src/s08_code_kickouts.py --video-id <id> --from-gt      # code at GT timestamps
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lib.config import load_config, resolve, out_dir, make_provenance
from lib.logging_setup import get_logger
from lib.pitch import Pitch
from lib.schema import read_table, write_table


def band_defenders(n: int, bands: list[list]) -> str:
    for lo, hi, label in bands:
        if lo <= n <= (hi if hi is not None else 10 ** 6):
            return label
    return "unknown"


def code_one(ev, tr: pd.DataFrame, teams: pd.DataFrame, cfg: dict, pitch: Pitch, log) -> dict:
    """Compute the paper's variables for one kickout."""
    C = cfg["coding"]
    att = cfg["pitch"]["attacking_end"]
    sign = 1 if att == "attacking" else -1
    line65 = pitch.line_x(65.0, att)
    line45 = pitch.line_x(45.0, att)

    # --- setup window: the formation is what it is just BEFORE the kick.
    # The paper counts players inside the 65 at the kickout, so sample a
    # short window ending at t_start rather than a single frame, which
    # would be at the mercy of one dropped detection.
    w0, w1 = ev.t_start_s - C["setup_window_s"], ev.t_start_s
    setup = tr[(tr.timestamp_s >= w0) & (tr.timestamp_s <= w1) & tr.pitch_x.notna()]
    peak = tr[(tr.timestamp_s >= ev.t_peak_s - C["peak_window_s"]) &
              (tr.timestamp_s <= ev.t_peak_s + C["peak_window_s"]) & tr.pitch_x.notna()]

    row: dict = {"event_id": ev.event_id, "t_start_s": float(ev.t_start_s),
                 "t_peak_s": float(ev.t_peak_s)}

    if setup.empty:
        row.update(codeable=0, reason_not_codeable="no registered tracks in setup window")
        return row

    # --- per-frame counts, then the median across the window ---------
    def counts(df, col_line):
        per = df.groupby(["frame_idx", "team"]).apply(
            lambda d: int((sign * d.pitch_x >= sign * col_line).sum()), include_groups=False)
        return per.unstack(fill_value=0) if len(per) else pd.DataFrame()

    c65 = counts(setup, line65)
    if c65.empty:
        row.update(codeable=0, reason_not_codeable="no team-assigned tracks in setup window")
        return row

    # ---- which team is defending? ---------------------------------
    # NOT "the team with more players inside the 65". That heuristic is
    # circular: at a kickout BOTH teams crowd the drop zone, and when the
    # defending side concedes (their smallest count, and the case the
    # paper reports as most distinctive) the attacking team has more
    # players there, so the heuristic inverts exactly when it matters.
    #
    # The kicking team is identified by its goalkeeper: the one player
    # standing deepest in their own half at the moment of the kick. The
    # defending team is the other one. Where the manual coding or the
    # fixture metadata already records who kicked out, that overrides —
    # in a real platform this is a field, not an inference.
    known = [c for c in c65.median().index if c != "unassigned"]
    if not known:
        row.update(codeable=0, reason_not_codeable="all tracks unassigned")
        return row

    override = getattr(ev, "kicking_team", None)
    if override in known:
        kicking = override
        row["team_source"] = "metadata"
    elif len(known) < 2:
        row.update(codeable=0, reason_not_codeable="only one team assigned in setup window")
        return row
    else:
        deepest = setup.loc[setup[setup.team.isin(known)].pitch_x.idxmin()] if sign > 0 \
            else setup.loc[setup[setup.team.isin(known)].pitch_x.idxmax()]
        kicking = deepest.team
        row["team_source"] = "goalkeeper_inferred"
        # Sanity: the keeper should be a long way behind everyone else.
        gap = abs(float(deepest.pitch_x) - float(
            setup[setup.team == kicking].pitch_x.quantile(0.1 if sign > 0 else 0.9)))
        row["keeper_gap_m"] = gap
        if gap < C["min_keeper_gap_m"]:
            row.update(codeable=0,
                       reason_not_codeable=f"cannot identify the kicking team "
                                           f"(deepest player only {gap:.1f} m behind the rest)")
            return row

    defending = [c for c in known if c != kicking]
    if not defending:
        row.update(codeable=0, reason_not_codeable="only the kicking team was assigned")
        return row
    defending, attacking = defending[0], kicking

    med = c65.median()
    n_def_65 = int(round(med.get(defending, 0)))
    n_unassigned_65 = int(round(med.get("unassigned", 0)))

    # --- nearest-opponent distance: the zonal / player-to-player test --
    nn_list = []
    for f, g in setup.groupby("frame_idx"):
        d_side = g[(g.team == defending) & (sign * g.pitch_x >= sign * line65)]
        a_side = g[g.team == attacking] if attacking else g.iloc[0:0]
        if len(d_side) < 3 or len(a_side) < 3:
            continue
        D = np.linalg.norm(d_side[["pitch_x", "pitch_y"]].values[:, None] -
                           a_side[["pitch_x", "pitch_y"]].values[None, :], axis=-1)
        nn_list.append(float(np.median(D.min(axis=1))))
    nn_opp_m = float(np.median(nn_list)) if nn_list else np.nan

    # --- the three defensive strategies, straight from their definitions
    if n_def_65 <= C["concede_max_defenders"]:
        strategy, why = "concede", f"{n_def_65} defenders inside the 65"
    elif np.isnan(nn_opp_m):
        strategy, why = "unclear", "too few registered players to measure marking distance"
    elif nn_opp_m <= C["player_to_player_max_nn_m"]:
        strategy, why = "player_to_player", f"median nearest-opponent {nn_opp_m:.1f} m"
    else:
        strategy, why = "zonal", f"median nearest-opponent {nn_opp_m:.1f} m"

    # --- short versus long, by where the ball is received -------------
    # Proxy for the reception point: the centroid of the tightest cluster
    # of players at the contest peak. Without ball tracking that is the
    # best available estimate, and its error is measured in S11.
    recv_x = np.nan
    if not peak.empty:
        pk = peak.groupby("track_id")[["pitch_x", "pitch_y"]].median()
        if len(pk) >= 2:
            P = pk.values
            D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
            np.fill_diagonal(D, np.inf)
            tightest = D.min(axis=1).argsort()[:C["landing_cluster_k"]]
            recv_x = float(P[tightest, 0].mean())
    distance_class = ("unclear" if np.isnan(recv_x)
                      else ("short" if sign * recv_x < sign * line45 else "long"))

    # --- contested proxy ---------------------------------------------
    contested_proxy = "unclear"
    if not peak.empty:
        pk = peak.groupby("track_id")[["pitch_x", "pitch_y"]].median().values
        if len(pk) >= 2:
            D = np.linalg.norm(pk[:, None] - pk[None, :], axis=-1)
            np.fill_diagonal(D, np.inf)
            contested_proxy = "contested" if D.min() <= C["contested_max_m"] else "uncontested"

    row.update(
        codeable=1, reason_not_codeable="",
        defending_team=defending, kicking_team=attacking,
        n_defenders_in_65=n_def_65,
        n_defenders_band=band_defenders(n_def_65, C["defender_bands"]),
        n_unassigned_in_65=n_unassigned_65,
        count_uncertainty=n_unassigned_65,
        nn_opponent_m=nn_opp_m,
        defensive_strategy=strategy,
        strategy_evidence=why,
        receive_x_m=recv_x,
        distance_class=distance_class,
        contested_proxy=contested_proxy,
        n_registered_setup=int(setup.track_id.nunique()),
        detector_source="",
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--from-gt", action="store_true",
                    help="code at manually coded timestamps, isolating coding from detection")
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg)
    log = get_logger("s08_code", od / "logs")
    pitch = Pitch(**cfg["pitch"]["dimensions"])

    tracks = read_table("tracks", od / "tracks.parquet")
    teams = read_table("track_teams", od / "track_teams.csv")

    # attach pitch coordinates and team to every track row
    import json
    from lib.pitch import apply_homography, foot_point
    Hs = json.loads((od / "homographies.json").read_text())
    shots = pd.read_csv(od / "shots.csv")
    shot_of = np.full(int(tracks.frame_idx.max()) + 1, -1, int)
    for r in shots.itertuples():
        shot_of[r.frame_start:r.frame_end + 1] = r.shot_id
    tracks = tracks.assign(shot_id=shot_of[tracks.frame_idx.values], pitch_x=np.nan, pitch_y=np.nan)
    foot = np.array([foot_point(*b) for b in tracks[["x1", "y1", "x2", "y2"]].values])
    for sid, g in tracks.groupby("shot_id"):
        H = Hs.get(str(int(sid)))
        if H is None:
            continue
        xy = apply_homography(np.array(H, float), foot[g.index.values])
        tracks.loc[g.index, ["pitch_x", "pitch_y"]] = xy
    tracks = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
    tracks["team"] = tracks.team.fillna("unassigned")

    if args.from_gt:
        events = pd.read_csv(resolve(cfg, "gt") / args.video_id / "gt_events.csv")
        events = events[events.coding_pass == 1]
        source = "gt_timestamps"
    else:
        events = read_table("events_pred", od / "events_pred.csv")
        source = "predicted_timestamps"
    log.info("coding %d kickouts from %s", len(events), source)

    rows = [code_one(ev, tracks, teams, cfg, pitch, log) for ev in events.itertuples()]
    out = pd.DataFrame(rows)
    out["detector_source"] = source
    write_table(out, "kickout_coding_pred", od / f"kickout_coding_{source}.csv", strict=False)

    ok = out[out.codeable == 1]
    log.info("%d of %d kickouts codeable (%.0f%%)", len(ok), len(out), 100 * len(ok) / max(len(out), 1))
    if len(out) > len(ok):
        log.info("not codeable, by reason:\n%s",
                 out[out.codeable == 0].reason_not_codeable.value_counts().to_string())
        log.info("COMPARE THIS TO THE PAPER: McColgan et al. excluded 29.4%% of kickouts "
                 "(908 of 3081) because broadcast camera angles and score replays hid them. "
                 "Your exclusion rate and theirs measure the same underlying problem from "
                 "opposite directions — that comparison is the strongest single result "
                 "available to this project.")
    if len(ok):
        log.info("defensive strategy distribution:\n%s",
                 ok.defensive_strategy.value_counts().to_string())
        log.info("defender band distribution:\n%s", ok.n_defenders_band.value_counts().to_string())
        log.info("median count uncertainty from unassigned players: +/- %.1f",
                 float(ok.count_uncertainty.median()))

    make_provenance("s08_code", cfg, [str(od / "tracks.parquet")],
                    dict(source=source, **cfg["coding"])).write(od / "provenance_s08.json")


if __name__ == "__main__":
    main()
