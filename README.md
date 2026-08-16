# gaa-kickout-vision

Can a published, expert-validated kickout coding scheme for Gaelic
football be reproduced automatically from broadcast video — and how much
of it survives the attempt?

---

## The premise

**McColgan, Bradley, Earle, Gaul & Martin (2026)**, *Identifying and
Defining Kickout Strategies in Senior Inter-County Ladies Gaelic
Football*, Applied Sciences 16(7):3277
([doi:10.3390/app16073277](https://doi.org/10.3390/app16073277), CC BY 4.0),
defined and validated seven kickout strategies, coded 2172 kickouts by
hand in NacSport across five championship seasons, and modelled which
strategies win possession.

Two facts from that paper define this project:

1. **They had to discard 908 of 3081 kickouts — 29.4% — because
   broadcast camera angles and score replays hid them.** The loss is
   non-random: stronger teams score more, scores trigger replays, replays
   hide the following kickout.
2. **Their conclusion calls for a central Gaelic Games video and tracking
   platform**, because the research is bottlenecked on the video rather
   than on the analysis.

This repository is a prototype of the computer-vision layer such a
platform would need, targeting the variable their own model says matters
most, and measuring honestly how far it gets.

---

## Status

Built and tested end to end; **not yet validated**. What is measured
today is the footage, not the model.

| Stage | State |
|---|---|
| s00–s03 ingest, prepare, detect, track | Running on 2 × 12 min windows |
| s04 pitch registration | Built, not run — needs hand-clicked landmarks |
| s05 features | Running, image and pitch space |
| s06 team assignment | Built, not run |
| s07 kickout localisation | Running; velocity term removed (see below) |
| s08–s11 coding and agreement | Built, blocked on ground truth |
| Box annotation | 2 × 60 s blocks drawn; **failed validation, being redrawn** |

### What has actually been established

Three measurements, all about the source rather than the method, and all
pointing the same way — broadcast footage is produced for viewing, not
for analysis.

**Frame-rate conversion upstream of delivery.** Roughly one frame in five
carries no new motion (repeat period ≈ 3.6 frames at 25 fps). Verified
against the raw file to rule out the resampling in `s01` — the raw source
has *more* low-difference frames than the working clip, so the conversion
diluted the pattern rather than creating it. This aliased against a
3-frame velocity window; widening past the repeat period cut
phase-dependence 7× (η² 0.00071 → 0.00010), and the per-phase drift went
from monotonic (+0.06 → −7.33 px/s) to essentially flat.

**Framing scale confounds any pixel-space motion feature.** Players
appear 2.3× smaller during kickouts than in open play, because the
broadcast pulls wide to frame a restart and tight to follow the ball. So
`mean_vertical_velocity` is *lower* at contests than elsewhere (ratio
0.62–0.89, never above 1) — the opposite of the leap signature it was
built to capture. The term was actively penalising true kickouts in the
candidate scorer and has been removed. Neither restricting to the nearest
players nor subtracting global camera motion recovers it: pure scaling
predicts 0.435 against 0.694 observed, so real contest motion is present
but swamped by a framing effect roughly twice its size. Logged as
`framing_scale` in `docs/10`, identified pre-audit by hypothesis testing
rather than retrofitted to explain observed errors.

**A one-frame offset in the evaluator**, from MOT's 1-based convention,
found and fixed. It would have shifted every box by 40 ms and looked like
a mediocre detector.

### Not yet measured

Precision, recall, mAP, and IDF1. The first
annotation pass failed its own validation: boxes were systematically too
loose (aspect h/w 1.63 against 2–3 for a standing player), which caps IoU
before the detector is involved, and recall rose 2.6× when the IoU
threshold was loosened — an annotation signature, not a detection one.
Re-annotating tight. **No detector number from this repo should be quoted
until that lands.**

---

## What it does

Designed to detect and track players, register each camera shot to pitch
coordinates, assign players to teams, and reproduce the paper's coding
variables in **metres rather than pixels** — because every one of their
definitions is anchored to a pitch line (players inside the opposition
65, kickouts received inside the 45, an opponent within 2 m, four players
across the 21).

| Target | Status | Why |
|---|---|---|
| **Players inside the opposition 65** | **Primary** | Strongest predictor in their mixed model; a counting task with no judgement, so ground truth is reliable |
| Defensive strategy (zonal / player-to-player / concede) | Secondary | Their definitions are geometric — three rules on nearest-opponent distance and defender count |
| Short vs long | Secondary | Reception point estimated from the landing cluster |
| Contested | Proxy only | "Player gaining possession" needs the ball; reported under a different name |
| Offensive strategies, outcome, shots | Out of scope | Depend on run intent, a keeper–outfield signal, or the ball |

Full rationale: [`docs/01_source_paper.md`](docs/01_source_paper.md).

---

## Results

Empty until the re-annotation lands. Fill from `outputs/<run_id>/tables/`.
See [`docs/09_results_template.md`](docs/09_results_template.md).

| | Value [95% CI] | n |
|---|---|---|
| Held-out pitch reprojection error (m) | | |
| Detection recall by player-size tercile | | |
| ICC(2,1), defenders inside the 65 | | |
| Exact-match rate, defender count | | |
| **Intra-rater ceiling (blind re-code)** | | |
| Defensive strategy exact agreement | | |
| **Coverage rate** (vs their 70.6% usable) | | |

---

## Quick start

```bash
pip install -r requirements.txt
make test          # known-answer statistics tests
make demo          # evaluation half, on synthetic data, no footage needed
```

With footage — see [`docs/14_build_schedule.md`](docs/14_build_schedule.md)
for the Tier 1 path, which skips automatic event detection and codes at
manually located timestamps, isolating coding accuracy from detection
accuracy:

```bash
python src/s00_ingest.py --file data/raw/match.mp4 --video-id lgf26_final_w1 ...
make tier1 VIDEO_ID=lgf26_final_w1
```

---

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| 00 | `s00_ingest.py` | Content hash, `ffprobe` spec, rights status → registry |
| 01 | `s01_prepare_footage.py` | Constant-frame-rate clip, frames, manifest |
| 02 | `s02_detect.py` | `detections.parquet` (YOLO, person class) |
| 03 | `s03_track.py` | `tracks.parquet` (ByteTrack) |
| **04** | **`s04_register_pitch.py`** | **Per-shot homography + held-out error in metres** |
| 05 | `s05_features.py` | Image features, then pitch features in metres |
| **06** | **`s06_assign_teams.py`** | **Team per track, with confidence and an unassigned class** |
| 07 | `s07_detect_kickouts.py` | Temporal localisation (Tier 2 — skippable) |
| **08** | **`s08_code_kickouts.py`** | **The paper's variables, per kickout** |
| 09 | `s09_evaluate_detection.py` | P/R/F1, mAP, recall by player size, MOTA/IDF1 |
| 10 | `s10_evaluate_events.py` | Temporal P/R/F1 at three tIoU thresholds |
| **11** | **`s11_evaluate_coding.py`** | **Per-variable agreement, the ceiling, the coverage rate** |
| 12 | `s12_failure_audit.py` | A clip per failure, for causal classification |
| 13 | `s13_render_overlay.py` | Overlay incl. pitch projection and a labelled failure |
| 14 | `s14_report.py` | Figures, markdown tables, `metrics.json` |
| 15 | `s15_review_tool.py` | Human-factors sub-study (optional) |
| — | `query.py` | DuckDB SQL over every artefact |

---

## Design notes

**Pitch registration is load-bearing, not polish.** No variable in the
source scheme is expressible in pixels. `s04` therefore reports
**held-out** reprojection error in metres, which is the floor on
everything downstream: at 2 m RMSE a player within 2 m of the 65 cannot
be classified, and the defender count carries that ambiguity explicitly.

**Team assignment fails openly.** Tracks whose kit colour does not
separate cleanly are marked `unassigned` rather than guessed, and the
count of unassigned players inside the 65 becomes a ± band on the primary
variable rather than invisible error.

**Time is defined once.** After `s01` the clip is constant-frame-rate and
`timestamp_s == frame_idx / fps` exactly. Container timestamps are never
read back.

**Schemas are enforced.** `schemas/tables.yaml` declares every column,
type, key and enum; each write is validated, so a malformed table fails
where it was produced.

**Failure is a value, not an exception.** `unassigned` is a legal team,
`unclear` a legal strategy, `codeable = 0` with a reason a legal row. The
coverage rate that falls out of those is a headline result, directly
comparable with the source paper's 29.4% exclusion rate, and it would be
destroyed by a pipeline that guessed rather than abstained.

**Statistics are implemented and tested.** The ICC in `src/lib/statsx.py`
matches `pingouin` to 1e-6; the suite pins the behaviours that are easy to
get wrong — one-to-one event matching, absolute-agreement ICC penalising
systematic bias, weighted kappa for the ordered band variable.

**Rejected approaches are recorded, not just the chosen one.** Repeat-frame
detection runs on pixels because the track-based alternative was tested
first and topped out at precision 0.40 / recall 0.43 — YOLO box
coordinates jitter ~1.4 px even on a repeated frame against ~4.9 px on a
normal one, and the distributions overlap too much. That rejection is in
the docstring so nobody retries it.

---

## Benchmarks — and how to quote them

The source paper reports inter-operator agreement of κ = 0.981–1.00 on
200 kickouts. **That is not a target for this system.** Their coders
classified kickouts a human had already located and tagged; this pipeline
must locate, register, detect and assign teams before it classifies, and
each stage adds error their figure never absorbed. Their κ benchmarks the
classification step in isolation. Report both framings and name which is
which.

---

## Known limitations

- **No validated detector numbers yet.** See Status above.
- Two 12-minute windows from a single match. Every interval will be wide.
  Feasibility study, not validation study.
- One coder; intra-rater reliability only, no inter-rater.
- Velocity is now phase-independent, **not absolutely correct**. The
  wider window roughly halved peak magnitudes (65 → 31 px/s). Downstream
  thresholding is percentile-based and therefore scale-invariant, but no
  absolute velocity from an earlier run is comparable.
- One homography per camera shot, so intra-shot pan and zoom is
  unmodelled drift — to be measured, not assumed away.
- Foot-point projection assumes ground contact, which is false for
  exactly the players who matter at an aerial contest.
- No ball tracking, so `contested` is a proxy under a different name.
- Camera-cut detection is **unresolved**. Mean absolute frame difference
  gives an answer that flips entirely with the threshold (1 cut per 2 s
  to 1 per 80 s across a plausible range), so cut rate is not reported.
  It needs hand-marked cuts or a scale-invariant measure.
- The paper is Ladies Gaelic Football; if your footage is men's GF, kick
  range and rules differ. Do not quote their frequencies as expectations.

---

## Data and rights

Footage is used **locally for method development** and is not
redistributed. No video, frames, or overlays are committed — enforced by
`.gitignore`, verified by `make check`. Derived artefacts hold box
coordinates, per-clip track indices and a binary team label only: no
identities, no biometric templates, no cross-clip re-identification.

`lgf26_final_w2` is held out as the only clean test set and has never
been inspected, tuned or thresholded on; the frame sampler hard-refuses
it by ID rather than relying on the operator.

The source paper is CC BY 4.0; its definitions and figures may be reused
with citation, and are cited wherever used.

---

## Documentation

[`docs/`](docs/README.md) — source paper and variable map, data collection
protocol, operational definitions, annotation SOP, data dictionary,
architecture, evaluation plan, statistical analysis plan, results
template, failure taxonomy, human factors, reproducibility, interview
notes, and the tiered build schedule.
