# 06 — Evaluation plan

Every metric this project reports, what it is computed on, and why it
earns its place. Metrics are grouped by the layer they evaluate, because
a failure at the detection layer and a failure at the event layer have
completely different implications and lumping them together hides that.

---

## 6.0 Pitch registration — the layer everything else rests on (`s04`)

| Metric | Definition | Report as |
|---|---|---|
| Reprojection RMSE, **held out** | Error in metres on landmarks excluded from the fit | median across shots + range |
| Reprojection RMSE, fit points | Same, on the fitting landmarks | For contrast only — always optimistic |
| Max held-out error | Worst landmark per shot | Single value |
| Shots registered / shots containing a kickout | Coverage of the registration step | Proportion |

**Report the held-out figure, never the fit figure.** A homography fitted
to eight points and evaluated on those eight points will look excellent
and mean nothing.

This number is the floor on every pitch-space variable, and it must be
carried into their interpretation explicitly: if held-out RMSE is 2 m,
then a player standing within 2 m of the 65 m line cannot be classified
reliably, and the defender count inherits that ambiguity. Report the
count alongside the number of players falling inside that band.

**Error propagation check worth running:** perturb each homography by its
own held-out error and recount the defenders inside the 65. The spread of
recounts is a direct, empirical uncertainty on the primary variable, and
it is a more convincing statement of uncertainty than any analytic one.

---

## 6.1 Detection — on the annotated frame subset (`s09`)

| Metric | Definition | Reported as |
|---|---|---|
| Precision | TP / (TP + FP) at IoU ≥ 0.5 | value + 95% Wilson CI |
| Recall | TP / (TP + FN) at IoU ≥ 0.5 | value + 95% Wilson CI |
| F1 | Harmonic mean of the above | value + propagated interval |
| mAP@0.5 | All-point interpolated AP, IoU 0.5 | single value |
| mAP@0.5:0.95 | COCO-style average over IoU thresholds | single value (optional) |
| **Recall by size tercile** | Recall within small / medium / large GT bbox-area terciles | 3 values + CIs |

Matching is greedy by descending IoU with a one-to-one constraint within
each frame. Terciles are computed on the **ground-truth** box area
distribution and their edges are reported in px², so the reader knows what
"small" means rather than trusting the word.

**The tercile row is the headline of this section.** A Gaelic football
pitch is 130–145 m long, longer than a soccer pitch, and a far-side
player in a 720p broadcast frame may be 15–25 px tall. Recall collapsing
in the small tercile is the quantitative form of "broadcast video cannot
see the far side of a GAA pitch", and it is the single most useful number
in the project for arguing that a fixed multi-camera rig is necessary.

Wilson intervals rather than Wald: at recall near 1.0 with small n, a Wald
interval extends above 1.0, which is embarrassing in a table.

---

## 6.2 Tracking — on the same annotated subset (`s06`)

| Metric | What it tells you | Why included |
|---|---|---|
| MOTA | Combined FP + FN + ID-switch penalty | Standard; dominated by detection, so never report alone |
| **IDF1** | Identity preservation across the sequence | **The one that matters here** — the features depend on tracks persisting |
| HOTA | Balances detection and association | Report if `trackeval` installs cleanly; skip rather than fight it |
| ID switches | Raw count | Interpretable, and quotable per-kickout |
| Fragmentations | How often a trajectory breaks | Directly measures the camera-cut problem |
| MT / ML | Mostly-tracked / mostly-lost proportions | Shows whether failure is spread across all players or concentrated in a few |

Computed with `motmetrics` at IoU 0.5. **Also report ID switches per
kickout contest**, not just in total: the claim being made is that
identity fails *specifically at the moment of interest*, and a
per-contest count is the evidence for it. A global count would let a
sceptic argue the switches happened during quiet passages.

---

## 6.3 Event detection — the headline result (`s07`)

| Metric | Definition |
|---|---|
| Precision / Recall / F1 at temporal IoU ≥ 0.5 | Greedy one-to-one matching, highest-confidence prediction first |
| Same at tIoU 0.3 and 0.7 | Shows sensitivity to boundary strictness |
| PR curve + AP | Sweep the confidence threshold; all-point interpolation |
| Mean/median absolute temporal offset | \|t_peak_pred − t_peak_gt\| over matched events |
| Signed mean offset | Distinguishes a **fixable systematic lag** from **unfixable scatter** |

The one-to-one constraint matters: without it, two overlapping
predictions on one real kickout both count as true positives and recall
is inflated. `tests/test_statsx.py::test_match_events_one_to_one` pins
this behaviour.

Reporting three tIoU thresholds is not padding. If F1 holds at 0.3 and
collapses at 0.7, the detector finds the right moments and gets the
boundaries wrong — a different problem, with a different fix, from a
detector that misses events entirely.

---

## 6.4 Team assignment (`s06`)

| Metric | Why |
|---|---|
| Proportion of tracks assigned | Unassigned players are missing from every count |
| Accuracy on a hand-labelled subset | Colour clustering is cheap and wrong sometimes; measure how often |
| Median count uncertainty per kickout | The +/- band the primary variable inherits |

Assignment errors do not cancel. A defender misassigned as an attacker
moves the count by two, not one, so the uncertainty band is roughly twice
the misassignment rate.

---

## 6.5 The coding scheme — the headline (`s11`)

One row per variable in the McColgan et al. (2026) scheme that the
pipeline attempts, each with the statistic appropriate to its
measurement level:

| Variable | Level | Statistics |
|---|---|---|
| **Players inside the opposition 65** | count | **ICC(2,1)**, Bland–Altman bias and LoA, exact-match rate, within-1 rate, mean absolute error |
| Defender band (0-4 / 5-7 / 8-10 / 11+) | ordinal | Linear **weighted** kappa, exact and within-one-band accuracy |
| Defensive strategy | nominal | Kappa, per-class recall, confusion matrix |
| Short vs long | binary | Kappa, precision, recall |
| Contested (proxy) | binary | Kappa — and a note that this is a different variable from theirs |

Unweighted kappa is wrong for the band variable: it treats confusing
0-4 with 11+ as no worse than confusing 8-10 with 11+, when the paper's
own model treats those bands as an ordered dose.

### Coverage — the number that connects back to the paper

| Metric | Definition |
|---|---|
| **Coverage rate** | Kickouts the pipeline could code / kickouts attempted |
| Exclusion reasons | Cross-tab: no registration, no team assignment, too few players detected, camera cut, replay |
| Published comparison | Their 908 of 3081 excluded (29.4%), for the same underlying reason |

This is the most quotable result in the project. Present it as a
trade-off, not a win: a human coding 70% of kickouts at near-perfect
agreement and a pipeline coding some other fraction at moderate agreement
are different points on a curve, and which is preferable depends on
whether you need one season or ten.

---

## 6.6 Agreement benchmarks, and how to quote them

Three reference points, in increasing order of relevance:

1. **Chance.** Baked into every kappa.
2. **Your intra-rater ceiling.** Blind re-code, 48 h apart. The ceiling
   for *this project's* labels.
3. **Their published inter-operator kappa (0.981–1.00, n=200 kickouts).**
   The ceiling for the *classification step under ideal conditions*.

Quoting (3) as a target for the end-to-end system would be a mistake an
examiner will spot immediately: their coders classified kickouts a human
had already located and tagged, with no detection, registration or team
assignment error upstream. Report the end-to-end number and the
classification-only number separately, and name which benchmark applies
to which.

---

## 6.7 Agreement with manual coding — statistical detail

This is the section that makes it a validation study rather than a demo.

| Metric | Applied to | Why this one |
|---|---|---|
| ICC(2,1) | Contest duration; `n_players_in_contest` | Two-way random effects, **absolute** agreement, single measure. Absolute because a model that is reliably 0.4 s early is not interchangeable with a human even at r = 1.0 |
| Bland–Altman bias + 95% LoA | Duration; `t_peak` offset | Gives the *range* of disagreement, not just its average. Direction convention: **model − human** |
| Proportional-bias regression | Difference on mean | If error grows with duration, a single LoA is the wrong summary — check before reporting one |
| Cohen's κ | Event presence per 5 s bin | Chance-corrected presence agreement, comparable to how reliability is reported in the sports-science literature |
| **Intra-rater κ and ICC** | Your pass 1 vs your blind pass 2 | **The ceiling.** Report beside every model score |

Model score alone: a number. Model score beside the human self-agreement
ceiling: a measurement with a scale. Always report both raw values; the
ratio is a summary, never a substitute.

---

## 6.8 Runtime and cost (`s02`, logged)

| Metric | Why it belongs in a research report |
|---|---|
| Inference throughput (fps), GPU and CPU | Determines whether real-time is even conceivable |
| GPU-seconds per minute of video | The cost model for scaling to a season |
| Wall-clock for the full pipeline on 10 min | Honesty about what "we could run this on everything" costs |

Extrapolate explicitly: seconds per video-minute × 70 min × N matches.
A supervisor building a platform cares about this number more than about
your F1, and almost no candidate provides it.

---

## 6.9 Human factors (`s15`, optional)

| Metric | Question it answers |
|---|---|
| Median seconds per event coded, manual vs assisted | Does the tool actually save time, or just move the work? |
| Acceptance rate by confidence band | Is the confidence score trustworthy enough to show a user? |
| Acceptance rate of model **errors** under assistance | Automation bias: does the tool make the human worse? |

Three numbers from one afternoon. The third is the interesting one: if an
analyst accepts false positives at a high rate, the system degrades the
analysis it was meant to support, and that is a human-factors result
rather than a CV result.

---

## What is deliberately *not* measured

Named so that their absence reads as a decision:

- **Ball detection accuracy** — no ball tracking, by design (§ charter).
- **Team or player identity** — out of scope, and a privacy question.
- **Pitch-space (metric) accuracy** — would require homography; all
  measurements are in pixels and are therefore camera-dependent. This is a
  real limitation and belongs in the write-up, not in a footnote.
- **Generalisation to other sports or camera setups** — n = 1–2 clips.
