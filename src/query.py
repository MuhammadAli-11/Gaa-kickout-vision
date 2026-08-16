#!/usr/bin/env python3
"""Retrieval layer — SQL over the parquet artefacts with DuckDB.

Small, but it is the piece that says 'data architecture' rather than
'script that prints numbers'. The tables are already columnar and
schema-checked, so a query engine over them is free, and it is the
natural seam where a real deployment would swap local parquet for object
storage plus a catalogue without any analysis code changing.

Examples:
    python src/query.py --sql "SELECT count(*) FROM tracks"
    python src/query.py --preset busiest_frames
    python src/query.py --preset contest_windows
"""
from __future__ import annotations

import argparse

import duckdb

from lib.config import load_config, out_dir

PRESETS = {
    "busiest_frames": """
        SELECT frame_idx, timestamp_s, n_tracks, centroid_spread, mean_vertical_velocity
        FROM features WHERE valid_frame = 1
        ORDER BY n_tracks DESC, centroid_spread ASC LIMIT 20""",
    "contest_windows": """
        SELECT e.event_id, e.t_start_s, e.t_end_s, e.confidence, e.n_players_in_contest,
               avg(f.centroid_spread) AS mean_spread,
               max(f.mean_vertical_velocity) AS peak_vertical_velocity
        FROM events_pred e JOIN features f
          ON f.timestamp_s BETWEEN e.t_start_s AND e.t_end_s
        GROUP BY ALL ORDER BY e.t_start_s""",
    "track_lifetimes": """
        SELECT track_id, count(*) AS n_frames,
               min(timestamp_s) AS first_seen, max(timestamp_s) AS last_seen,
               max(timestamp_s) - min(timestamp_s) AS lifetime_s, avg(bbox_area) AS mean_area
        FROM tracks GROUP BY track_id ORDER BY lifetime_s DESC LIMIT 25""",
    "detection_density": """
        SELECT floor(timestamp_s / 30) * 30 AS half_minute,
               count(*) / 30.0 AS dets_per_second, avg(conf) AS mean_conf
        FROM detections GROUP BY 1 ORDER BY 1""",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sql")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    od = out_dir(cfg, create=False)
    con = duckdb.connect()
    for name, fn in [("detections", "detections.parquet"), ("tracks", "tracks.parquet"),
                     ("features", "features.parquet")]:
        if (od / fn).exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{od / fn}')")
    for name, fn in [("events_pred", "events_pred.csv"), ("event_matches", "event_matches.csv")]:
        if (od / fn).exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{od / fn}')")

    sql = args.sql or PRESETS[args.preset or "contest_windows"]
    print(con.execute(sql).df().head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
