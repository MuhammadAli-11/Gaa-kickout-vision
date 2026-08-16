# Where this repo ends and Project 2 begins

**Project 1 — `gaa-kickout-vision`** (this repo): the vision layer.
Detection, tracking, pitch registration, team assignment, reproduction of
the published kickout coding scheme, and the validation that says how far
it gets.

**Project 2 — `gaa-video-environment`**: the data environment. Ingestion,
storage, catalogue, retrieval, access policy, orchestration.

The two overlap, because a vision pipeline with no data discipline is a
pile of scripts. Rather than duplicate that work, this repo contains a
*minimal, working* version of the Project 2 concerns, built to the scale
of one match. When you build Project 2 properly, these are the pieces
that migrate out and grow up:

| In this repo | Project 2 version |
|---|---|
| `src/s00_ingest.py` — SHA-256, `ffprobe`, rights status, one CSV registry | Ingest service; object storage keyed by content hash; Postgres or Iceberg catalogue |
| `schemas/tables.yaml` + `src/lib/schema.py` — validated table contracts | Versioned schema registry with a `schema_version` column and migrations |
| `src/lib/config.py` — `run_id` derived from a config hash, provenance JSON per stage | Lineage tracking across a DAG runner with caching and retries |
| `src/query.py` — DuckDB views over local parquet | The same SQL against object storage plus a catalogue; analysis code unchanged |
| `rights_status` as a registry column | Access policy enforced at query time |
| `data/gt/` conventions | Annotation service with multiple coders and inter-rater reliability built in |

The seam is deliberate: nothing in `s02`–`s11` reads across videos, so
scaling from one match to a season is a scheduling problem, not a
rewrite. That property is worth stating out loud — it is the difference
between a prototype that can grow and one that has to be thrown away.

**For the interview:** if asked why a vision project contains an
ingestion registry, the answer is that the source paper's problem was
never the model. McColgan et al. lost 29.4% of their kickouts to the
video, and their conclusion asks for a platform. Project 1 measures the
loss; Project 2 removes it.
