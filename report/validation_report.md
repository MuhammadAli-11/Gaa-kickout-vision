# Automating a published kickout coding scheme from broadcast video: a feasibility and validation study

**Author:** ____________  **Date:** ____________
**Code:** `github.com/<you>/gaa-kickout-vision`  **Run:** `run_________`

---

## 1. Aim

McColgan et al. (2026) defined and validated a coding scheme for kickout
strategy in Ladies Gaelic Football, applied it by hand to 2172 kickouts,
and discarded 908 more — 29.4% — because broadcast footage did not show
them; their conclusion calls for a central video and tracking platform to
remove that constraint. This study asks how much of their scheme can be
produced automatically from the same kind of footage. It targets the
variable their model identifies as the strongest predictor of kickout
outcome — the number of players committed inside the opposition 65 —
measures agreement against manual coding using their published
definitions, establishes the ceiling via a blind re-code, and reports the
coverage rate as the direct counterpart to their exclusion rate.
**This is a feasibility study on a small sample, not a validation study.**

## 2. Method

**Footage.** ___ minutes of broadcast Gaelic football (____________,
_______), standardised to 25 fps and 1280×720 with a constant frame rate,
so that `timestamp_s = frame_idx / 25` holds exactly.

**Pipeline.** YOLOv8m person detection (imgsz 1280, conf 0.25) →
ByteTrack association → per-shot homography from hand-clicked pitch
landmarks, giving player positions in metres → team assignment by torso
colour clustering, with an explicit `unassigned` class → the paper's
variables computed over a 1.5 s window ending at the kick.

**Pitch registration.** ___ shots registered from ___ containing a
kickout. Median **held-out** reprojection error ___ m (range ___–___).
Pitch modelled as ___ m × ___ m.

**Operational definitions.** Taken verbatim from Table 1 of McColgan et
al. (2026) (CC BY 4.0). Temporal boundaries, which their NacSport
workflow did not require, were added and are flagged as additions in
`docs/03`, together with every point at which the automated computation
necessarily departs from their definition.

**Ground truth.** ___ kickouts coded by one analyst using their
definitions (`gt_coding.csv`, pass 1); a random 25% of the timeline
re-coded blind ≥ 48 h later (pass 2).

___ frames were annotated with boxes in MOT16 format for detection and
tracking metrics.

**Analysis.** Pre-specified in `docs/07` before results were inspected.
Proportions with Wilson intervals; F1 with a percentile bootstrap over
events; ICC(2,1) for continuous agreement; Bland–Altman for bias and
limits of agreement; Cohen's κ for presence in 5 s bins.

## 3. Results

*(Paste from `outputs/<run_id>/tables/`. Do not retype.)*

**Detection** (___ annotated frames, ___ boxes)

<!-- t1_detection.md -->

Recall stratified by ground-truth bounding-box area:

<!-- t2_size.md -->

**Tracking**

<!-- t3_tracking.md -->

**Pitch registration**

<!-- t0_registration.md -->

**Team assignment** — ___% of tracks assigned; median count uncertainty
± ___ players per kickout.

**PRIMARY: players inside the opposition 65**

<!-- t6c_defender_count.md -->

Errors large enough to move a kickout across one of the paper's band
boundaries (0-7 / 8-10 / 11+): ___ of ___.

**Defensive strategy**

<!-- t6d_strategy.md -->

**Coverage — the comparison with the source paper**

| | This pipeline | McColgan et al. (2026) |
|---|---|---|
| Kickouts available | | 3081 |
| Coded | | 2172 |
| **Excluded / uncodeable** | | **908 (29.4%)** |

**Agreement, and the two ceilings**

<!-- t6_agreement.md -->

Their published inter-operator agreement (κ = 0.981–1.00, n = 200) is
reported here for reference only. It benchmarks classification of
human-located kickouts, not end-to-end automated coding, and is not a
target for this system.

**Figures.** Fig 1 PR curve (n = ___ events, tIoU 0.5). Fig 2
Bland–Altman for duration and peak time (difference = model − human).
Fig 3 recall by size tercile with Wilson intervals. Fig 4 feature series
with manual coding and predictions marked. Fig 5 failure causes.

**Runtime.** ___ fps on ____________; ___ GPU-seconds per video-minute;
extrapolating, ___ for a 70-minute match and ___ for a 30-match season.

## 4. Limitations

*This is the longest section, and deliberately so.*

**Sample size.** With n = ___ events, the 95% interval on event F1 runs
from ___ to ___. Differences smaller than approximately 0.25 in
proportion are not detectable. No conclusion here rests on a comparison
finer than that.

**The label ceiling.** Intra-rater κ was ___ [___, ___] and ICC on
duration ___ [___, ___]. The model reaches ___ of that ceiling. Where
the ceiling is low, the binding constraint is the operational definition
and the difficulty of coding broadcast footage, not the model.

**Single coder.** Inter-rater reliability cannot be computed. Intra-rater
agreement is an optimistic proxy: a coder is more consistent with
themselves than two coders are with each other.

**Pixel space.** No homography was estimated, so all spatial features are
in image coordinates and depend on camera position and zoom. "Within 5 m
of the landing point" was judged by eye against pitch markings.

**Registration error propagates.** With a median held-out error of ___ m,
a player within ___ m of the 65 m line cannot be classified reliably; a
median of ___ players per kickout fell inside that band. Perturbing each
homography by its own error and recounting gave a spread of ___ players
on the primary variable.

**One homography per shot.** Broadcast cameras pan and zoom continuously
within a shot; measured drift over 3 s was ___ m.

**Foot-point projection assumes contact with the ground**, which is false
for exactly the players who matter at an aerial contest.

**The six failure modes.**

1. *Aerial occlusion* (___ cases). Bodies overlap at the contest;
   detection merges or drops boxes precisely when the event occurs.
2. *Camera pan and cut* (___ cases). Broadcast direction follows the
   ball, so tracks break at the moment of interest. Broadcast direction
   is optimised for viewing, not for analysis.
3. *Scale* (___ cases). Recall on the smallest bbox-area tercile was
   ___ against ___ for the largest. A GAA pitch is 130–145 m long; at
   720p a far-side player may be 15–25 px tall.
4. *No ball* (___ cases). Without ball tracking there is no "player
   gaining possession", so the contested variable is a proxy reported
   under a different name.
5. *Registration failure* (___ cases). Shot too tight, or too few pitch
   lines visible, to solve a homography.
6. *Team assignment failure* (___ cases). Kit colour did not separate at
   the available resolution.

<!-- t7_failures.md -->

**Annotation error.** ___ cases were errors in the ground truth rather
than the model. These were logged, not silently corrected, and give a
direct empirical estimate of label noise.

**Structural artefact.** In rule mode the detector emits fixed-length
windows, so predicted contest duration is constant and ICC on duration is
structurally near zero irrespective of model quality. Duration agreement
is therefore reported as *not assessable* in this configuration rather
than as a model result.

**Replays.** Broadcast footage contains slow-motion replays of the events
being detected. Replays were excluded from ground truth and detections on
them counted as false positives, on the reasoning that a deployed system
ingesting a live feed would face the same problem.

## 5. What I would do next

- **Fixed multi-camera capture at one venue** — the platform the source
  paper's own conclusion calls for. It removes registration as a per-shot
  problem (calibrate once), removes camera cuts entirely, and directly
  addresses failure modes 1, 2, 3 and 5, which together account for ___%
  of uncodeable kickouts. A broadcast feed is optimised for an audience,
  and no amount of modelling recovers information the director chose not
  to show.
- **Ball tracking, or an event feed for disambiguation.** Failure mode 4
  is the measured cost of the no-ball design; this quantifies what
  removing it would buy.
- **A larger annotated corpus with two or more independent coders.**
  Achieving a ±0.15 interval on κ needs on the order of 100+ scored bins
  and 50+ events, well beyond a 10-minute clip. Multiple coders replace
  an intra-rater proxy with a real inter-rater reliability estimate.

*(Optional fourth, if the human-factors sub-study was run.)*

- **A properly powered analyst-in-the-loop study.** On n = 1, verifying a
  candidate took ___ s against ___ s to code from scratch, and ___% of
  model false positives were accepted. Whether an assisted workflow
  improves accuracy per unit time — or merely moves the work — is the
  question that decides whether any of this gets used.


## References

McColgan, A.; Bradley, J.; Earle, D.; Gaul, D.; Martin, D. (2026).
Identifying and Defining Kickout Strategies in Senior Inter-County Ladies
Gaelic Football. *Applied Sciences* 16(7), 3277.
https://doi.org/10.3390/app16073277 (CC BY 4.0)
