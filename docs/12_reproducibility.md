# 11 — Reproducibility

## Environment

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version        # system binary, required by s00/s01/s09
```

Record and paste into the report: Python version, OS, GPU model and
driver, `ultralytics` version, and the YOLO weights file used with its
SHA-256. Model weights change between releases; "yolov8m" alone is not a
specification.

```bash
python -c "import ultralytics, torch, sys; print(sys.version); print(ultralytics.__version__, torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
sha256sum yolov8m.pt
```

## Determinism

`set_seeds()` fixes `random`, `numpy`, and `torch` from
`project.seed`. Two caveats to state rather than paper over:

- **GPU inference is not bit-deterministic** across driver or hardware
  changes. Detection counts can vary by a handful of boxes. Everything
  downstream of `detections.parquet` is fully deterministic.
- **Bootstrap intervals are seeded** and will reproduce exactly given the
  same input table.

## Run identity

`run_id` = short hash of the `video`, `detect`, `track`, `features` and
`events` config blocks. Outputs live in `outputs/<run_id>/`, so:

- Re-running with identical config overwrites the same directory (idempotent).
- Changing any parameter produces a new directory; nothing is silently lost.
- The directory name identifies the configuration that produced every
  number inside it.

Each stage also writes `provenance_sNN.json` with `git_sha`,
`config_hash`, inputs, parameters and timing.

## Full re-run

```bash
make all VIDEO_ID=nfl25_r3_h2
```

or explicitly:

```bash
python src/s00_ingest.py --file data/raw/source.mp4 --video-id nfl25_r3_h2 ...
python src/s01_prepare_footage.py --video-id nfl25_r3_h2
python src/s02_detect.py          --video-id nfl25_r3_h2
python src/s03_track.py           --video-id nfl25_r3_h2
python src/s04_features.py        --video-id nfl25_r3_h2
python src/s05_detect_events.py   --video-id nfl25_r3_h2
python src/s06_evaluate_detection.py --video-id nfl25_r3_h2
python src/s07_evaluate_events.py    --video-id nfl25_r3_h2
python src/s08_agreement.py          --video-id nfl25_r3_h2
python src/s09_failure_audit.py      --video-id nfl25_r3_h2
python src/s10_render_overlay.py     --video-id nfl25_r3_h2
python src/s11_report.py             --video-id nfl25_r3_h2
```

## Reproducing without footage

The evaluation half runs end to end on synthetic data, which is how the
statistics are tested and how the pipeline can be demonstrated on a laptop
with no video present:

```bash
make demo        # synthetic tracks + ground truth -> s04..s08, s11
pytest tests/ -q # 13 known-answer tests, incl. ICC cross-checked vs pingouin
```

This matters for the interview: it means the statistical machinery is
validated against data with a known answer, independently of whether the
detector works.

## What is committed and what is not

**Committed:** all code, `config.yaml`, `schemas/tables.yaml`, docs, the
report, generated figures, `metrics*.json`, the tables in `outputs/*/tables/`.

**Never committed:** video, extracted frames, overlay renders, failure
clips, parquet artefacts, model weights. Enforced by `.gitignore`; verify
with `git status` before pushing.

Consequence to state in the README: the repository is **not** independently
reproducible by a third party, because the footage cannot be shared. It is
reproducible by anyone holding the source file with the recorded SHA-256.
Say this plainly — the alternative is a reader assuming it is reproducible
and discovering otherwise.
