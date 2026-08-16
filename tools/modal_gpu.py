#!/usr/bin/env python3
"""Run the two GPU stages (s02 detect, s03 track) on Modal.

Everything else in this pipeline is CPU work and stays local: ffmpeg
standardisation, features, team assignment, the statistics, the failure
audit, the report. Only s02 and s03 put YOLO over all 18,000 frames.

The remote side is a faithful copy of the repo layout, because
lib/config.py sets REPO_ROOT from its own location (parents[2]) and every
path resolves from there. Code lands at /root/repo, so REPO_ROOT is
/root/repo and cfg['paths'] resolves exactly as it does on your machine.

ONE WINDOW PER INVOCATION. s02 writes outputs/<run_id>/detections.parquet
and s03 writes tracks.parquet, neither keyed by video_id — so two windows
sharing a run_id would overwrite each other. Run w1, pull it down, then
run w2.

Usage:
    pip install modal && modal setup

    # 1. push the clip (once per window)
    modal volume create gaa-kickout-data
    modal volume put gaa-kickout-data \
        data/clips/lgf26_final_w1_working.mp4 clips/lgf26_final_w1_working.mp4

    # 2. run both GPU stages
    modal run tools/modal_gpu.py --video-id lgf26_final_w1

    # 3. pull results into the local run directory
    modal volume get gaa-kickout-outputs run_4c9cad33 outputs/
"""
from __future__ import annotations

import modal

REPO = "/root/repo"

app = modal.App("gaa-kickout-gpu")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # opencv needs these even in headless builds
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.120",
        "torch==2.13.0",
        "torchvision==0.28.0",
        "opencv-python-headless>=4.9",
        "pandas>=2.2",
        "pyarrow>=16.0",
        "pyyaml>=6.0",
        "numpy>=1.26",
    )
    # Bake the weights into the image so they are not re-downloaded on every
    # run, and so a cold container is not racing the network before inference.
    .run_commands(
        "mkdir -p /weights",
        'cd /weights && python -c "from ultralytics import YOLO; YOLO(\'yolov8m.pt\')"',
    )
    # Ultralytics writes a settings file; keep it off the read-only layers.
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics"})
    .add_local_dir("src", f"{REPO}/src")
    # schemas/tables.yaml is NOT optional: lib/schema.py resolves it from
    # REPO_ROOT and every stage validates through write_table(), so without
    # it s02 runs the full 18,000 frames of inference and then dies on the
    # write. Anything else lib/ reads from REPO_ROOT belongs here too.
    .add_local_dir("schemas", f"{REPO}/schemas")
    .add_local_file("config.yaml", f"{REPO}/config.yaml")
)

# Files the stages read from REPO_ROOT that are not produced at runtime.
# Checked before inference so a packaging slip costs a second, not an hour.
REQUIRED_FILES = ["config.yaml", "schemas/tables.yaml"]

# Clips in, outputs out. Kept separate so the ~380 MB clip is uploaded once
# and never re-downloaded when you fetch results.
data_vol = modal.Volume.from_name("gaa-kickout-data", create_if_missing=True)
out_vol = modal.Volume.from_name("gaa-kickout-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={f"{REPO}/data": data_vol, f"{REPO}/outputs": out_vol},
    timeout=60 * 60 * 2,
)
def run_stages(video_id: str, stages: list[str]) -> dict:
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    os.chdir(REPO)

    missing = [f for f in REQUIRED_FILES if not (Path(REPO) / f).exists()]
    if missing:
        raise SystemExit(
            f"missing from the image: {missing} — add them with "
            f"add_local_dir/add_local_file in the image definition above"
        )

    clip = Path(REPO) / "data" / "clips" / f"{video_id}_working.mp4"
    if not clip.exists():
        raise SystemExit(
            f"no clip at {clip} — upload it first:\n"
            f"  modal volume put gaa-kickout-data "
            f"data/clips/{video_id}_working.mp4 clips/{video_id}_working.mp4"
        )

    # config's detect.weights is a bare filename, resolved relative to cwd.
    weights = Path(REPO) / "yolov8m.pt"
    if not weights.exists():
        shutil.copy("/weights/yolov8m.pt", weights)

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    scripts = {"s02": "src/s02_detect.py", "s03": "src/s03_track.py"}
    for st in stages:
        if st not in scripts:
            raise SystemExit(f"unknown stage {st!r}; expected s02 and/or s03")
        print(f"\n=== {st} :: {video_id} ===", flush=True)
        subprocess.run(
            [sys.executable, scripts[st], "--video-id", video_id],
            check=True,
        )

    # Without an explicit commit the writes stay in the container.
    out_vol.commit()

    produced = sorted(
        str(p.relative_to(Path(REPO) / "outputs"))
        for p in (Path(REPO) / "outputs").rglob("*")
        if p.is_file()
    )
    print("\nproduced:", *produced, sep="\n  ")
    return {"video_id": video_id, "files": produced}


@app.local_entrypoint()
def main(video_id: str, stages: str = "s02,s03") -> None:
    result = run_stages.remote(video_id, [s.strip() for s in stages.split(",")])
    print(f"\ndone: {result['video_id']}")
    print("fetch with:  modal volume get gaa-kickout-outputs <run_id> outputs/")
