#!/usr/bin/env python3
"""S00 — Ingestion. Register source footage with provenance before touching it.

This stage exists because the studentship is about data ARCHITECTURE as
much as vision. Nothing enters the pipeline without: a content hash, a
rights status, and a probed technical spec. If a result later looks odd,
the registry says exactly which file, which codec and which frame rate
produced it.

Usage:
    python src/s00_ingest.py --file data/raw/match.mp4 --video-id gaa_lf_r3 \
        --competition "Allianz League Div 1" --match-date 2025-02-16 \
        --broadcaster "TG4" --rights personal_local_only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lib.config import load_config, resolve, make_provenance
from lib.logging_setup import get_logger
from lib.schema import write_table


def sha256_of(path: Path, nbytes: int = 0, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            read += len(b)
            if nbytes and read >= nbytes:
                break
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    meta = json.loads(subprocess.check_output(cmd, text=True))
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return dict(
        duration_s=float(meta["format"].get("duration", 0.0)),
        src_fps=round(fps, 4),
        src_width=int(v["width"]),
        src_height=int(v["height"]),
        codec=v.get("codec_name", "unknown"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--file", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--competition", default="")
    ap.add_argument("--round", dest="round_or_stage", default="")
    ap.add_argument("--match-date", default="")
    ap.add_argument("--broadcaster", default="")
    ap.add_argument("--capture-source", default="broadcast",
                    choices=["broadcast", "club_fixed_cam", "handheld", "other"])
    ap.add_argument("--rights", default="personal_local_only",
                    choices=["personal_local_only", "licensed", "public_domain", "unknown"])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("s00_ingest")
    src = Path(args.file).resolve()
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    log.info("probing %s", src.name)
    tech = ffprobe(src)
    log.info("hashing (%s)", cfg["ingest"]["hash_algorithm"])
    digest = sha256_of(src, int(cfg["ingest"]["hash_bytes"]))

    row = dict(
        video_id=args.video_id, sha256=digest, filename=src.name,
        bytes=src.stat().st_size, **tech,
        competition=args.competition, round_or_stage=args.round_or_stage,
        match_date=args.match_date, broadcaster=args.broadcaster,
        capture_source=args.capture_source, rights_status=args.rights,
        ingest_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=args.notes,
    )

    reg_path = resolve(cfg, "raw") / "video_registry.csv"
    df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
    df = pd.concat([df[df.get("video_id", pd.Series(dtype=str)) != args.video_id],
                    pd.DataFrame([row])], ignore_index=True)
    write_table(df, "video_registry", reg_path)

    if tech["src_fps"] and abs(tech["src_fps"] - cfg["video"]["fps"]) > 0.01:
        log.warning("source is %.3f fps, pipeline standardises to %d fps — s01 will resample",
                    tech["src_fps"], cfg["video"]["fps"])
    if tech["src_width"] < cfg["video"]["width"]:
        log.warning("source is narrower (%dpx) than the working resolution — upscaling adds no information",
                    tech["src_width"])

    make_provenance("s00_ingest", cfg, [str(src)], row).write(
        resolve(cfg, "interim") / args.video_id / "provenance_s00.json")
    log.info("registered %s (%s...) -> %s", args.video_id, digest[:12], reg_path)


if __name__ == "__main__":
    main()
