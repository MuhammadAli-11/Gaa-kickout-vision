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

## What it does

Detects and tracks players, registers each camera shot to pitch
coordinates, assigns players to teams, and reproduces the paper's coding
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

## Results

The primary variable is **not yet measured**: it needs `s04` pitch
registration, which needs hand-clicked landmarks. What follows is what has
actually been measured, with the parts that are not yet trustworthy said
plainly. Empty tables live in
[`docs/09_results_template.md`](docs/09_results_template.md).

### What the footage does to the measurement

Three findings, reached by three unrelated routes, all saying the same
thing: **broadcast footage is optimised for viewing, not for analysis.**
Full write-ups in [`docs/10_failure_taxonomy.md`](docs/10_failure_taxonomy.md).

| Finding | Measurement | Consequence |
|---|---|---|
| **Frame-rate conversion upstream** | 19.1% of raw 28 fps frames near-identical to their predecessor, modal gap 4 frames — measured on the source, before `s01` | Velocity depended on repeat phase. Fixed by widening the difference window past the repeat period: η² 0.00071 → 0.00010 |
| **`framing_scale`** | Contest framing puts players at **0.435×** their open-play apparent size (110.7 px vs 254.4 px mean bbox height) | Pixel velocity is *lower* at contests than in open play (ratio 0.694). The `leap` term was scoring kickouts backwards |
| **Cuts are not separable from pans** | Top inter-frame differences in the annotated blocks reach only 1.1–1.3× p99 | No hard cut can be confirmed without a human watching. `annotation.blocks.confirmed_cuts_s` stays empty rather than thresholded |

### Kickout candidate detection (`s07`, rule mode)

A measurement artefact found and corrected. `leap` rewarded high vertical
velocity; because of `framing_scale` that feature is *lowest* at real
contests, so the term penalised the windows it was meant to promote.
Reproduce either row via `events.rule.terms`.

| | 4-term (with `leap`) | 3-term (removed) |
|---|---|---|
| Kickouts recovered, ±3 s | 4 / 8 | **7 / 8** |
| Median \|offset\| of matches | 1.56 s | **0.80 s** |
| Median score rank of a true kickout | 443 / 1494 | **223 / 1494** |
| Precision@8 | 1 / 8 | 1 / 8 |

Recall roughly doubles and the rank of a true kickout halves — but
precision@8 is unchanged, so what was fixed is a term pointing the wrong
way, **not** the detector's discriminative power, which remains poor.
n = 8 rough timestamps, not coded ground truth.

### Detection and tracking (`s09`) — measured, not yet trustworthy

Two contiguous 60 s blocks of `lgf26_final_w1`, 546 annotated frames,
4,384 hand boxes. Wilson intervals.

| | Value [95% CI] |
|---|---|
| Precision @ IoU 0.5 | 0.134 [0.125, 0.143] |
| Recall @ IoU 0.5 | 0.185 [0.174, 0.197] |
| F1 @ IoU 0.5 | 0.155 [0.146, 0.165] |
| mAP@0.5 | 0.054 |
| IDF1 / MOTA | 0.147 / −0.705 |

> **Do not quote these as detector performance.** They are bounded by
> annotation box tightness, not by the model. GT boxes have aspect ratio
> h/w **1.63** against the detector's **2.16** — on the same players, hand
> boxes are ~2× too wide and ~1.3× too tall. Recall is 2.6× higher at IoU
> 0.3 (0.473) than at 0.5 (0.185), which is the signature of loose boxes
> rather than missed players. The size-tercile table is likewise
> uninterpretable: recall is *highest* on small boxes, because terciles cut
> on GT area and the loosest boxes land in "large". `tools/validate_annotation.py`
> fails this annotation on two structural errors as well. The pretrained
> baseline is currently **unmeasured**, not bad.

### Annotation

<!-- Replace the two files below with the real figures; captions and
     filenames are already wired up so nothing else needs editing. -->

![Annotated blocks: ground truth against detector output](docs/img/fig_annotation_blocks.svg)

*Figure 1 — Hand annotation against YOLOv8m output on `blk_01` (420–480 s)
and `blk_02` (600–660 s).*

![Box-size distributions, annotation against detector](docs/img/fig_annotation_size_gap.svg)

*Figure 2 — Box area and aspect ratio, 4,384 hand boxes against 6,056
detections on the same 546 frames.*

Conventions used when drawing them:
[`docs/img/annotation_conventions.svg`](docs/img/annotation_conventions.svg).

### Throughput and cost

Measured on Modal, A10G, 2026-08-15: `s02` 69.0 fps, `s03` 48.2 fps,
**52.9 GPU-seconds per video-minute** combined. `s02` on CPU is
ffmpeg-bound (85.1% of wall clock) rather than model-bound (12.5%), so the
GPU buys less than the fps figure suggests.

### Still blocked

| | Blocked on |
|---|---|
| Pitch registration, all metre-space variables | Hand-clicked landmarks (`s04 --annotate`) |
| **Players inside the opposition 65** (primary) | The above |
| Event-level P/R, temporal localisation | A real `gt_events.csv`; only the rough-timestamp skeleton exists |
| Failure audit cross-tab | The above, plus human classification — `cause` is empty by design |
| Trustworthy detection/tracking metrics | Re-annotation with tight boxes |

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
python src/s00_ingest.py --file data/raw/match.mp4 --video-id lgf25_r3 ...
make tier1 VIDEO_ID=lgf25_r3
```

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

Supporting tools:

| Tool | Purpose |
|---|---|
| `tools/sample_frames_for_annotation.py` | Chooses what to annotate and records why. `--mode blocks` (contiguous, interpolable, yields track IDs) or `--mode sparse` |
| `tools/validate_annotation.py` | **Run before trusting any metric.** Structural checks on the labels themselves: MOT16 parse, frame alignment, track-ID usability, continuity, box geometry |
| `tools/extract_segments.py` | Merges padded kickout windows out of an edited compilation |
| `tools/modal_gpu.py` | Runs `s02`/`s03` on a Modal GPU by subprocessing the existing scripts, never reimplementing them |

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

**Statistics are implemented and tested.** The ICC in `src/lib/statsx.py`
matches `pingouin` to 1e-6; the suite pins the behaviours that are easy to
get wrong — one-to-one event matching, absolute-agreement ICC penalising
systematic bias, weighted kappa for the ordered band variable.

## Benchmarks — and how to quote them

The source paper reports inter-operator agreement of κ = 0.981–1.00 on
200 kickouts. **That is not a target for this system.** Their coders
classified kickouts a human had already located and tagged; this pipeline
must locate, register, detect and assign teams before it classifies, and
each stage adds error their figure never absorbed. Their κ benchmarks the
classification step in isolation. Report both framings and name which is
which.

## The split — binding

| Video | Role |
|---|---|
| `lgf26_final_w1` | Development. Annotate, diagnose, tune |
| **`lgf26_final_w2`** | **Never inspected.** The only clean test set left |
| `data_2.mp4` (compilation) | Dropped as a tuning source — editing is a confound |

`lgf26_final_w2` is listed in `annotation.holdout_video_ids`, and the
sampler hard-refuses that id rather than trusting the operator to pass the
right flag. Every number above comes from `w1`.

## Known limitations

- **The current annotation is not usable for metrics.** Boxes are
  systematically loose; `blk_01` is additionally missing 54 of 300
  keyframes and has 20% of rows exported as CVAT shapes rather than
  tracks, which carry no identity. Re-annotation with tight boxes would
  move the numbers more than any model change.
- Two 60 s blocks and one validation block is a thin base. With two blocks
  the only honest split is 1 train / 1 val, so validation is a single 60 s
  passage.
- Velocity is phase-*independent* after the repeat-frame fix, not correct
  in absolute terms — the window is 3× wider, so peak magnitudes are
  roughly halved. Absolute velocities from earlier runs are not comparable.
- `scene_change_flag` is a Jaccard distance over track-ID sets, tuned to
  plausibility and **never validated against hand-marked cuts**. It is an
  identity-churn proxy, not a cut detector.
- n ≈ 15 kickouts from one clip. Every interval is wide. Feasibility
  study, not validation study.
- One coder; intra-rater reliability only, no inter-rater.
- One homography per camera shot, so intra-shot pan and zoom is
  unmodelled drift — measured, not assumed away.
- Foot-point projection is wrong for a leaping player, which is precisely
  what happens at a contest.
- **Pixel-space motion features are confounded by framing.** Because the
  broadcast changes focal length between contest and open play, no
  pixel-velocity feature can be compared across windows. The fix is metres
  via `s04`, which is unproven here and may not survive ~2 m hold-out
  reprojection error on an airborne player's foot point.
- No ball tracking, so `contested` is a proxy under a different name.
- The paper is Ladies Gaelic Football; if your footage is men's GF, kick
  range and rules differ. Do not quote their frequencies as expectations.

## Data and rights

Footage is used locally for method development and is not redistributed.
No video, frames, or overlays are committed — enforced by `.gitignore`,
verified by `make check`. Derived artefacts hold box coordinates,
per-clip track indices and a binary team label only: no identities, no
biometric templates, no cross-clip re-identification.

The source paper is CC BY 4.0; its definitions and figures may be reused
with citation, and are cited wherever used.

## Documentation

[`docs/`](docs/README.md) — source paper and variable map, data
collection protocol, operational definitions, annotation SOP, data
dictionary, architecture, evaluation plan, statistical analysis plan,
results template, failure taxonomy, human factors, reproducibility,
interview notes, and the tiered build schedule.
