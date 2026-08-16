# 08 — Results template

Empty tables to fill from `outputs/<run_id>/tables/*.md`. Do not retype
numbers; `s11_report.py` generates every table as markdown so the report
cannot drift from the run that produced it.

Alongside each table is a **plausible range** and a **red flag**. If a
result lands in the red-flag column, something is wrong with the setup,
not brilliant about the model — check before celebrating.

---

## R1. Corpus description

| | Value |
|---|---|
| Clips | |
| Total duration analysed | |
| Frames | |
| Kickout contests coded (pass 1) | |
| Contests re-coded (pass 2) | |
| Frames with box annotation | |
| Ground-truth boxes | |
| Mean contest duration (SD) | |

## R1b. Pitch registration — the floor under everything

| Metric | Value | Plausible | Red flag |
|---|---|---|---|
| Shots registered / shots with a kickout | | | < 0.7 → most kickouts are uncodeable |
| Median **held-out** reprojection RMSE (m) | | 0.5–2.0 | **< 0.2** → you are reporting fit error, not hold-out |
| Max held-out error (m) | | | |
| Intra-shot drift over 3 s (m) | | 0.5–3.0 | 0 → check you actually re-registered |
| Players per kickout within RMSE of the 65 m line | | | The ambiguity band on the primary variable |

## R2. Detection (`t1_detection.md`)

| Metric | Value [95% CI] | Plausible | Red flag |
|---|---|---|---|
| Precision @ IoU 0.5 | | 0.75–0.92 | |
| Recall @ IoU 0.5 | | 0.60–0.85 | |
| F1 @ IoU 0.5 | | 0.70–0.88 | |
| mAP@0.5 | | 0.70–0.90 | **> 0.95** → evaluating on frames the detector saw, or annotation copied from predictions |

## R3. Recall by player size (`t2_size.md`) — **the key limitation table**

| Tercile | n GT boxes | Recall [95% CI] | Plausible | Red flag |
|---|---|---|---|---|
| Small | | | 0.35–0.65 | **> 0.80** → check tercile edges; boxes may all be similar size |
| Medium | | | 0.65–0.85 | |
| Large | | | 0.80–0.95 | |

Report the tercile edges in px²: small < ______, medium < ______.

## R4. Tracking (`t3_tracking.md`)

| Metric | Value | Plausible | Red flag |
|---|---|---|---|
| MOTA | | 0.35–0.65 | |
| IDF1 | | 0.40–0.65 | **> 0.85** → the clip is probably static; check for camera motion |
| ID switches (total) | | | **0** → tracking is not running |
| **ID switches per contest** | | 3–15 | |
| Fragmentations | | | |
| MT / ML | | | |

## R5. Event detection (`t4_events.md`) — **headline**

| tIoU | TP | FP | FN | Precision [CI] | Recall [CI] | F1 [CI] |
|---|---|---|---|---|---|---|
| 0.30 | | | | | | |
| 0.50 | | | | | | |
| 0.70 | | | | | | |

AP (tIoU 0.5): ______

Plausible F1 @ 0.5: 0.50–0.75. **> 0.90 → almost certainly leakage, or too
few events for the number to mean anything.** State n every time this
number appears.

## R6. Temporal localisation (`t5_offset.md`)

| | Value | Plausible | Red flag |
|---|---|---|---|
| Mean \|offset\| | | 0.5–2.0 s | **< 0.2 s** → suspicious; check for label leakage |
| Median \|offset\| | | | |
| Signed mean | | | |
| 95th percentile \|offset\| | | | |
| n matched | | | |

Interpretation to write: a consistent signed offset is a **systematic lag**
and is fixable with a constant correction; a near-zero signed mean with a
large absolute mean is **scatter**, and is not.

## R6b. Team assignment

| | Value | Red flag |
|---|---|---|
| Tracks assigned to a team | | < 60% → report total, not per-team, counts |
| Assignment accuracy on hand-labelled subset | | |
| Median count uncertainty per kickout (± players) | | > 3 → the primary variable is not usable |

## R6c. **PRIMARY — players inside the opposition 65**

| Metric | Value [95% CI] | n |
|---|---|---|
| ICC(2,1), automated vs manual | | |
| Bland–Altman bias (players, model − human) | | |
| 95% limits of agreement | | |
| Exact match rate | | |
| Within ±1 player | | |
| Mean absolute error | | |
| **Intra-rater ceiling, ICC** | | |

Interpretation to write: state how often the count error was large enough
to move the kickout across one of the paper's band boundaries (0-7 /
8-10 / 11+), because that is the error that would change their modelled
effect on kickout outcome. An ICC of 0.8 with errors that never cross a
band is a more useful system than an ICC of 0.85 with errors that do.

## R6d. Defensive strategy

| | Value [95% CI] | n |
|---|---|---|
| Exact agreement | | |
| Cohen's κ | | |
| Recall, zonal | | |
| Recall, player-to-player | | |
| Recall, concede | | |

Confusion matrix, and the threshold sweep on nearest-opponent distance.
If most errors are zonal ↔ player-to-player, that is one threshold, not a
broken method — show the curve.

## R6e. **COVERAGE — the comparison with the source paper**

| | This pipeline | McColgan et al. (2026) |
|---|---|---|
| Kickouts available | | 3081 |
| Kickouts coded | | 2172 |
| **Excluded / uncodeable** | | **908 (29.4%)** |
| Reason: camera angle / cut | | part of the 29.4% |
| Reason: replay | | part of the 29.4% |
| Reason: registration failure | | n/a (manual coding) |
| Reason: team assignment failure | | n/a |
| Reason: too few players detected | | n/a |

Their exclusions were also non-random — stronger teams score more, scores
trigger replays, replays hide the following kickout. State whether yours
show the same skew.

## R7. Agreement and the ceiling (`t6_agreement.md`) — **the distinctive table**

| Metric | Model vs human | **Human vs self (ceiling)** |
|---|---|---|
| Cohen's κ, presence (5 s bins) | | |
| ICC(2,1), contest duration | | |
| ICC(2,1), n players in contest | | |
| Bland–Altman bias, duration (s) | | |
| 95% LoA, duration (s) | | |
| Bias, t_peak (s) | | |
| 95% LoA, t_peak (s) | | |

Also report: p₀, pₑ and prevalence for each κ; proportional-bias slope and
p for each Bland–Altman.

> **Check before filling the duration row.** In `rule` mode the detector
> emits fixed-length windows, so predicted duration is constant and ICC is
> structurally ~0 whatever the model does. Either implement boundary
> refinement first, or write "not assessable in rule mode" in the cell.
> An ICC of 0 reported as a model finding is a mistake a supervisor will
> spot immediately — catching it yourself is worth more than the metric.

## R8. Failure audit (`t7_failures.md`)

| Cause | FP | FN | Boundary | Total | % |
|---|---|---|---|---|---|
| aerial_occlusion | | | | | |
| camera_pan_cut | | | | | |
| scale_distance | | | | | |
| no_ball_ambiguity | | | | | |
| annotation_error | | | | | |
| other | | | | | |

## R9. Runtime and cost

| | GPU | CPU |
|---|---|---|
| Detection throughput (fps) | | |
| Full pipeline wall-clock, 10 min footage | | |
| GPU-seconds per video-minute | | — |
| **Extrapolated: one 70-min match** | | |
| **Extrapolated: a 30-match season** | | |

## R10. Human factors (optional, `review_log.csv`)

| | Manual | Model-assisted |
|---|---|---|
| Median seconds per event coded | | |
| Events coded in 20 min | | |
| Acceptance rate, conf < 0.4 | — | |
| Acceptance rate, conf 0.4–0.7 | — | |
| Acceptance rate, conf > 0.7 | — | |
| **Acceptance rate of model false positives** | — | |

## Figures

| Figure | File | Caption must state |
|---|---|---|
| 1 | `fig1_pr_events.png` | n events, tIoU threshold |
| 2 | `fig2_bland_altman.png` | Difference direction (**model − human**), n, LoA values |
| 3 | `fig3_recall_by_size.png` | Tercile edges in px², n per tercile |
| 4 | `fig4_feature_timeline.png` | Green = manual coding, red = prediction |
| 5 | `fig5_failure_causes.png` | n cases classified |
