# 00 — Project charter

## The claim being tested

> The kickout coding scheme published by McColgan et al. (2026) — built
> by hand in NacSport, at the cost of discarding 29.4% of all kickouts
> because broadcast footage did not show them — can be partially
> reproduced automatically from the same broadcast footage. The agreement
> can be quantified variable by variable, the coverage rate can be
> compared directly with their exclusion rate, and the conditions under
> which broadcast video defeats the method can be characterised.

Note what is *not* claimed. Not that it works well. Not that it is
deployable. The deliverable is a measured answer with an honest error
budget, and a characterisation of the conditions under which broadcast
video makes the task impossible.

## Why this is the right project for this studentship

The advertised position has four named strands. A pure detection demo
addresses one of them. This project is scoped so that each strand has
something concrete attached to it:

| Studentship strand | What in this repo speaks to it |
|---|---|
| **Computer vision** — the visual intelligence layer | `s02`–`s08`: detection, tracking, pitch registration, team assignment, and reproduction of the published coding variables |
| **Data architecture** — ingestion, storage, retrieval, AI-first | `s00` registry with content hashing and rights status; `schemas/tables.yaml` as an enforced contract; columnar parquet artefacts; `src/query.py` SQL retrieval layer; per-stage provenance JSON |
| **Human factors** — systems that work under match-day pressure | `s15` analyst-in-the-loop study: time-per-event coded with and without model assistance, acceptance rate by confidence band, automation-bias check |
| **Applied data engineering** — research to deployment | Deterministic `run_id` from config hash, schema validation at every write, idempotent stages, runtime and cost-per-video-minute measured in `s02` |

The single-sentence pitch:

> *I automated a piece of coding your group currently does by hand,
> measured the agreement against my own manual coding, established the
> ceiling by measuring how well I agree with myself, characterised the
> four conditions under which broadcast video defeats the method, and
> measured whether an analyst is actually faster with the tool than
> without it.*

## Scope boundaries, stated up front

**In scope:** person detection, tracking, scene-level behavioural
features, temporal localisation of contests, full validation, failure
characterisation, a small human-factors measurement.

**Out of scope, deliberately:** ball detection and tracking; team
identification and jersey colour classification; player identity; homography
to pitch coordinates; kickout outcome classification; anything requiring
more than one camera. Each of these is named in the write-up as future
work with a reason, because a scope boundary you can defend reads as
judgement, and one you cannot reads as an oversight.

## Success criteria

The project succeeds if, at the end, all five of these exist:

1. Agreement on the primary variable — players inside the opposition 65 —
   **with an interval, an n, and the homography error it rests on**.
2. An intra-rater ceiling to compare that number against, plus an honest
   statement of why the paper's published inter-operator kappa of
   0.981–1.00 is not a like-for-like benchmark.
3. **A coverage rate**, stated beside their 29.4% exclusion rate.
4. A cross-tabulated failure audit with every uncodeable kickout assigned
   a cause.
5. A 2-minute overlay showing the pitch-space projection, including at
   least one labelled failure.
6. A 2-page report where the limitations section is the longest.

It does **not** succeed by producing high agreement. High agreement on 15
kickouts with no interval is a worse outcome than moderate agreement that
is properly bounded and whose error sources are traced.
