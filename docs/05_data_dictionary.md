# 04 — Data dictionary

`schemas/tables.yaml` is the authoritative, machine-readable version;
`src/lib/schema.py` validates every write against it, so a stage cannot
emit a table a later stage cannot read. This page is the human rendering.

Conventions used throughout:

- **Time** is `timestamp_s`, float seconds from the start of the working
  clip, always equal to `frame_idx / fps`.
- **Coordinates** are pixels in the standardised 1280×720 frame, origin
  top-left, `y` increasing downward. Vertical *velocity* is negated so
  that upward motion is positive, because "player leaps" reading as a
  negative number causes sign errors in every plot.
- **No identities** are stored anywhere: `track_id` is an arbitrary
  integer scoped to one clip and is not stable across runs.

---

## Provenance and registry

### `video_registry.csv` — produced by `s00_ingest.py`
The index of every source file. One row per video, keyed on `video_id`.

| Column | Type | Notes |
|---|---|---|
| `video_id` | string | Stable short id, appears in every downstream path |
| `sha256` | string | Content hash — answers "which file produced this number" |
| `filename`, `bytes` | string, int | As found on disk |
| `duration_s`, `src_fps`, `src_width`, `src_height`, `codec` | — | Probed with `ffprobe`, **not** assumed |
| `competition`, `round_or_stage`, `match_date`, `broadcaster` | string | Context for the write-up |
| `capture_source` | enum | `broadcast` / `club_fixed_cam` / `handheld` / `other` |
| `rights_status` | enum | `personal_local_only` / `licensed` / `public_domain` / `unknown` |
| `ingest_utc` | string | ISO 8601 |

### `provenance_sNN.json` — one per stage
`stage`, `run_id`, `git_sha`, `config_hash`, `created_utc`, `inputs`,
`params`. Written beside every artefact. This is what makes a number in
the report traceable to the code and configuration that produced it.

---

## Pipeline artefacts

### `manifest.csv` — `s01`
`frame_idx` (0-based, contiguous), `timestamp_s`, `path`, `width`,
`height`, `video_id`. The definition of the time axis.

### `detections.parquet` — `s02`
`frame_idx`, `det_idx`, `timestamp_s`, `x1`, `y1`, `x2`, `y2`, `conf`,
`class_id`. Raw and **unfiltered** above the low inference threshold, so
that thresholds can be swept without re-running inference.

### `tracks.parquet` — `s03`
`track_id`, `frame_idx`, `timestamp_s`, box corners, `cx`, `cy`,
`bbox_h`, `bbox_area`, `conf`. Tracks shorter than
`track.min_track_len_frames` are dropped; the count dropped is logged, not
hidden.

### `features_image.parquet` — `s05 --image-only`
One row per frame, in pixels. Drives shot segmentation and Tier 2
temporal detection. Columns: `n_tracks`, `cluster_density_px`,
`centroid_spread_px`, bbox heights, `mean/max_vertical_velocity`,
`scene_change_flag`, `valid_frame`.

### `pitch_keypoints.csv` — human, via `s04 --annotate`
`shot_id`, `landmark`, `img_x/img_y` (px), `pitch_x/pitch_y` (m),
`use_for_fit`. The hand-clicked correspondences.

### `homographies.csv` / `.json` — `s04 --fit`
Per shot: `n_landmarks`, `rmse_fit_m`, **`rmse_holdout_m`**,
`max_holdout_m`, `validated`. **Report the hold-out column.** It is the
floor on every pitch-space variable.

### `track_teams.csv` — `s06`
`track_id`, `team` in {team_a, team_b, **unassigned**}, `confidence`,
`n_crops`, `method`. Unassigned is a legitimate value; the number of
unassigned players inside the 65 becomes the ± band on the primary
variable.

### `features_pitch.parquet` — `s05`
One row per frame, in **metres**. The layer that makes the paper's
scheme computable.

| Column | Units | Meaning and why it is here |
|---|---|---|
| `n_players_registered` | count | Players successfully projected this frame |
| **`n_inside_65`** | count | **The paper's key defensive count** |
| `n_inside_45` | count | Supports the short/long split |
| `mean_x_m` | m | How far up the pitch the group sits |
| `spread_x_m`, `spread_y_m` | m | Dispersion, zoom-invariant unlike the pixel version |
| `width_used_m`, `depth_used_m` | m | Effective playing space the formation occupies |
| `nn_distance_m` | m | Mean nearest-neighbour distance — the marking signal |
| `cluster_density_m` | m | Mean distance to the k nearest players |
| `mean_speed_ms`, `max_speed_ms` | m/s | Real speed, not pixel speed |
| `frac_above_sprint` | 0–1 | Proxy for the "hard run" separating Purposeful from Limited Movement. **A proxy, and labelled as one** |
| `shot_id` | — | Which homography produced these metres |

### `events_pred.csv` — `s07` (Tier 2)
`event_id`, `t_start_s`, `t_end_s`, `t_peak_s`, `duration_s`,
`confidence`, `n_players_in_contest`, `detector_mode`.

> **Known structural limitation.** In `rule` mode every predicted event
> inherits the fixed analysis-window length, so predicted `duration_s` is
> constant. ICC on duration is then structurally ~0 regardless of model
> quality, because a constant has no variance to correlate. Either add
> boundary refinement (expand from `t_peak_s` until `centroid_spread`
> recovers to its local baseline) or state explicitly that duration
> agreement is not assessable in rule mode. **Do not report an ICC of 0
> as if it were a finding about the model.** This is exactly the kind of
> artefact worth catching and saying out loud.

### `kickout_coding_*.csv` — `s08`
The automated reproduction of the paper's variable set, one row per
kickout: `n_defenders_in_65` (primary), `n_defenders_band`,
`defensive_strategy`, `distance_class`, `contested_proxy`,
`nn_opponent_m`, `count_uncertainty`, `codeable`,
`reason_not_codeable`, `team_source`, `keeper_gap_m`.

`codeable = 0` rows are the coverage denominator and are as important as
the coded ones.

### `gt_coding.csv` — human, see docs/03 and docs/04
The manual counterpart, coded using the paper's Table 1 definitions.
Includes a `confidence` column for your own coding, so agreement can be
split by label quality.

### `gt_events.csv` — human, see docs/04
`event_id`, `t_start_s`, `t_end_s`, `t_peak_s`, `n_players_in_contest`,
`outcome`, `restart_type`, `coder_id`, `coding_pass` (1 or 2),
`visibility`, `notes`.

### `gt_boxes.txt` — human, MOT16
`frame` (1-based), `id`, `bb_left`, `bb_top`, `bb_width`, `bb_height`,
`conf`, `class`, `visibility`.

---

## Evaluation artefacts

| File | Produced by | Contents |
|---|---|---|
| `event_matches.csv` | `s07` | One row per prediction and per unmatched GT: `kind` (TP/FP/FN), `tiou`, peak times, durations. The input to the failure audit |
| `pr_curve_events.csv` | `s07` | Recall, precision, confidence — the PR curve |
| `metrics_detection.json` | `s06` | P/R/F1, mAP, recall by size tercile, MOTA/IDF1 block |
| `metrics_events.json` | `s07` | Per-tIoU metrics with intervals, AP, temporal offsets |
| `metrics_coding.json` | `s11` | Per-variable agreement, **intra-rater ceiling**, **coverage rate vs the paper's 29.4% exclusion** |
| `metrics.json` | `s11` | All of the above plus the config that produced them |
| `failure_audit.csv` | `s09` + human | `case_id`, `error_type`, `t_center_s`, `clip_path`, `cause` |
| `review_log.csv` | `s12` | Analyst verdicts, decision times, condition |
