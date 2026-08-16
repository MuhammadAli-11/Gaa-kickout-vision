# 03 — Operational definitions

**Do not write your own.** Use the definitions published in Table 1 of
McColgan et al. (2026), which were developed over five stages, circulated
to five senior inter-county coaches and four goalkeepers for scrutiny,
revised after a validation panel, and re-circulated for agreement. That
process is worth more than anything you could produce in a week, and
using it means a disagreement between your coding and theirs is a real
disagreement rather than a difference of vocabulary.

The paper is CC BY 4.0, so reproduce their definitions verbatim, cite the
DOI beside them, and code against those exact words.

---

## 1. What to copy across

Open the paper's Table 1 and copy the following rows into
`data/gt/_templates/coding_manual.md`, unaltered:

**Event and outcome** — Kickout; Kickout Won; Kickout Lost; Short
Kickout; Long Kickout; Contested Kickout; Uncontested Kickout; Time
Taken for Kickout.

**Defensive strategy** — Zonal; Player to Player; Concede; and the
player-count bands measured inside the opposition 65 m line.

**Offensive strategy** — Bunch and Break; Overload; Flat 4; Purposeful
Movement; Limited Movement. Copy these even though the pipeline does not
attempt them: you will code them by hand, and having them recorded lets
you report which offensive strategies your false-positive contests
actually were.

Also reproduce the nine-zone map from their Figure 1 into
`docs/img/zones.png` before using `Pitch.zone_of()`. A zone grid that
does not match theirs makes any zone comparison meaningless.

## 2. What you must add, because the paper did not need it

Their coders worked in NacSport on kickouts a human had already found.
You are locating kickouts in time, so you need temporal boundaries they
never had to define. These are **yours**, and must be flagged as
additions rather than presented as part of their scheme:

| Boundary | Definition | Note |
|---|---|---|
| `t_start_s` | First frame in which the ball has visibly left the goalkeeper's hand or foot | Precision: nearest frame (0.04 s). Never round to whole seconds |
| `t_peak_s` | First frame in which any player makes contact with the ball, or the ball lands untouched | The point at which the contest resolves |
| `t_end_s` | First frame of clean possession, or the ball becoming dead | Clean possession = two hands, control, plus one further step or a completed hand-pass |
| `setup window` | The 1.5 s ending at `t_start_s` | The window over which the formation is read. The pipeline uses the same window — see `coding.setup_window_s` |

## 3. Where the automated version necessarily departs

Log every one of these in the report. A silent departure is the
difference between a validation study and a misleading one.

| Their definition | What the pipeline can actually compute | Consequence |
|---|---|---|
| Players inside the opposition 65 | Same, given a homography and team assignment | Faithful, subject to registration error and unassigned players |
| Player-to-player: defenders stay within about 1 m of an opponent | Median nearest-opponent distance below a threshold | A distributional summary, not a per-player rule. Sweep the threshold and report the curve |
| Zonal: players occupy zones inside the opposition 65 | Not player-to-player, and more than the concede count | Defined by exclusion. State that |
| Contested: an opponent within 2 m of the player gaining possession | Minimum inter-player distance in the landing cluster at the peak | **Different variable.** No ball, so no "player gaining possession". Named `contested_proxy` in every table for exactly this reason |
| Short: received inside the 45 | Cluster-centroid estimate of the reception point relative to the 45 | Inherits both registration error and reception-point error |
| Time taken: ball dead to the kick | Not attempted in the core build | Requires detecting the ball going dead, a separate problem |

## 4. Coding procedure

1. Code `gt_events.csv` first — the temporal boundaries above.
2. Then code `gt_coding.csv` — the paper's variables, one row per
   kickout, using their definitions and the pitch markings visible in the
   frame. Set `coding_pass = 1` and `coder_id` to your initials.
3. Record `confidence` (high / medium / low) per row. Low-confidence
   rows are where the model is most likely to "disagree" with a label
   that was itself a guess, and separating those out is the honest way to
   interpret the agreement statistics.
4. Wait 48 hours, then blind re-code a random 25% (see docs/04 §C).

## 5. Edge cases

Append as you hit them. The number of rows here is itself a measure of
how ill-posed the task is on broadcast footage.

| Timestamp | Situation | Decision | Consistent with their Table 1? |
|---|---|---|---|

Two that will definitely come up:

**Replays.** The paper's largest exclusion category. Decide now:
replays are excluded from ground truth, and a detection on a replay is a
false positive — because a deployed system ingesting a live feed faces
exactly that. Log every replay timestamp; the count is a result.

**Partially visible formations.** If the camera does not show the full
press, you cannot count defenders inside the 65 and neither could their
coders. Code it `unclear` and let it fall into the exclusion rate rather
than guessing. The exclusion rate is a headline number in this project,
so contaminating it with guesses destroys the comparison with theirs.
