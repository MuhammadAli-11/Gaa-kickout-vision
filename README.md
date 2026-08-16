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

Fill from `outputs/<run_id>/tables/`. See
[`docs/09_results_template.md`](docs/09_results_template.md).

| | Value [95% CI] | n |
|---|---|---|
| Held-out pitch reprojection error (m) | | |
| ICC(2,1), defenders inside the 65 | | |
| Exact-match rate, defender count | | |
| **Intra-rater ceiling (blind re-code)** | | |
| Defensive strategy exact agreement | | |
| **Coverage rate** (vs their 70.6% usable) | | |

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

## Known limitations

- n ≈ 15 kickouts from one clip. Every interval is wide. Feasibility
  study, not validation study.
- One coder; intra-rater reliability only, no inter-rater.
- One homography per camera shot, so intra-shot pan and zoom is
  unmodelled drift — measured, not assumed away.
- Foot-point projection is wrong for a leaping player, which is precisely
  what happens at a contest.
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
