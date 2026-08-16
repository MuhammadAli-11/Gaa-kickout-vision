# 05 — Pipeline architecture

## Stage graph

```
                    data/raw/<video>.mp4
                            |
   [ s00_ingest ]      hash + ffprobe + rights   -> video_registry.csv
                            |
   [ s01_prepare ]     ffmpeg CFR 25fps 1280x720 -> working.mp4, frames/, manifest
                            |
   [ s02_detect ]      YOLO person detection     -> detections.parquet
                            |
   [ s03_track ]       ByteTrack association     -> tracks.parquet
                            |
   [ s05 --image-only ] image-space features     -> features_image.parquet
                            |                       (shot segmentation)
   [ s04_register ]    HAND-CLICKED landmarks    -> homographies.json + error in METRES
                            |                       <-- the load-bearing stage
   [ s05_features ]    pitch-space features (m)  -> features_pitch.parquet
                            |
   [ s06_teams ]       jersey colour clustering  -> track_teams.csv (+ 'unassigned')
                            |
   [ s07_detect_ko ]   temporal localisation     -> events_pred.csv   (Tier 2, skippable)
                            |                       or use manual timestamps (Tier 1)
   [ s08_code ]        THE PAPER'S VARIABLES     -> kickout_coding_*.csv
                            |
        +---------------+---+-----------+---------------+
        |               |               |               |
   s09_eval_det    s10_eval_events  s11_eval_coding  s12_failure_audit
   (boxes/tracks)  (temporal)       (HEADLINE)       (causes)
        |               |               |               |
        +---------------+-------+-------+---------------+
                                |
                        [ s14_report ] -> figures/, tables/, metrics.json
                                |
             [ s13_overlay ] -> overlay.mp4 with pitch projection + a labelled failure
             [ s15_review  ] -> review_log.csv (human factors, optional)
             [ query.py    ] -> DuckDB SQL over every artefact
```

Sixteen stages plus a retrieval utility. Note the loop at the top: `s05`
runs twice, before and after registration, because shot segmentation is
needed to know *what* to register and registration is needed to produce
metres. That dependency is real, not an artefact of the layout.

### The critical path

`s04` is the bottleneck and the only stage requiring human input mid-pipeline.
Everything downstream of it is in metres and everything upstream is in
pixels. If registration fails for a shot, every kickout in that shot is
uncodeable — which is not a bug to route around but the measurement the
project exists to make.

## Design decisions worth defending out loud

**The pipeline works in metres, not pixels, from s04 onward.** This is
the single most consequential architectural decision, and it was forced
by the source paper: every variable in the McColgan et al. (2026) scheme
is defined against a pitch line or an inter-player distance in metres. A
pixel-space pipeline cannot express any of them. So registration is a
first-class stage with its own error metric rather than a preprocessing
detail, and that error propagates explicitly into the primary variable.

**Failure is a value, not an exception.** `unassigned` is a legal team.
`unclear` is a legal strategy. `codeable = 0` with a reason is a legal
row. The coverage rate that falls out of those is a headline result,
directly comparable with the source paper's 29.4% exclusion rate, and it
would be destroyed by a pipeline that guessed rather than abstained.

**Stage boundaries are file boundaries.** Each stage writes a durable
artefact rather than passing objects in memory. Costs a little disk;
buys the ability to re-run evaluation forty times without re-running
inference once, which is what makes a four-day timeline feasible.

**Columnar storage for the per-frame tables.** `detections`, `tracks` and
`features` are parquet: typed, compressed, and directly queryable by
DuckDB with no load step. The event-level tables are CSV because they are
small and a human edits them. Format follows access pattern, not habit.

**Schemas are enforced, not documented.** `schemas/tables.yaml` declares
columns, types, primary keys, enums and ranges; `write_table()` validates
before writing. A malformed table fails at the stage that produced it,
not three stages downstream.

**Run identity is derived, not assigned.** `run_id` is a hash of the
config sections that affect the result. Change the confidence threshold
and outputs land in a new directory; re-run with identical settings and
they land in the same one. Results cannot be silently overwritten, and
"which config produced Table 3" is answerable from the directory name.

**Provenance travels with the artefact.** Every stage writes `git_sha`,
`config_hash`, inputs, timing, and parameters beside its output.

**Retrieval is a seam, not a feature.** `src/query.py` registers the
parquet files as DuckDB views. Locally that is a convenience. The point is
architectural: analysis code talks SQL to a catalogue, so swapping local
parquet for object storage plus a table catalogue changes one function and
no analysis. That is the migration path from a laptop to a research
platform, and it is worth being able to sketch on a whiteboard.

## What would change at real scale

Honest answers to the obvious follow-up, "what if it were 500 matches?"

| Concern | This project | At scale |
|---|---|---|
| Storage | Local disk | Object store, content-addressed by SHA-256 (already computed at ingest) |
| Catalogue | `video_registry.csv` | Postgres or Iceberg table; same columns |
| Orchestration | Numbered scripts, `make` | A DAG runner with retries and caching, same stage boundaries |
| Compute | One GPU, sequential | Shard by video, embarrassingly parallel — no stage crosses videos |
| Schema evolution | Edit `tables.yaml` | Versioned schema, `schema_version` column on every table |
| Pitch registration | Hand-clicked, per shot | Fixed cameras calibrated once per venue; the per-shot problem disappears |
| Team identity | Colour clustering per clip | Roster and kit reference held as fixture metadata, so `kicking_team` is a field rather than an inference |
| Access control | Local only | Rights status is already a column; it becomes a policy filter |

The reason the table is short is that the stage boundaries were drawn
with it in mind. Nothing in `s02`–`s08` reads across videos, so
parallelising is a scheduling problem rather than a rewrite.
