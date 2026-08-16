# Documentation index

**Start with [01_source_paper.md](01_source_paper.md).** Everything else
follows from it: this project automates a published, expert-validated
coding scheme rather than one invented for the occasion, and that choice
determines the target variables, the ground truth, the benchmark, and
the reason the pipeline works in pitch metres instead of pixels.

| Doc | Purpose | Done by |
|---|---|---|
| [00_project_charter.md](00_project_charter.md) | The claim, and how it maps to the studentship's four strands | Wed |
| **[01_source_paper.md](01_source_paper.md)** | **The paper, the variable map, what is and is not attempted** | **Wed** |
| [02_data_collection_protocol.md](02_data_collection_protocol.md) | Sourcing, registry, storage, rights | Wed |
| [03_operational_definitions.md](03_operational_definitions.md) | Their definitions, your additions, and where the automation departs | Wed |
| [04_annotation_guide.md](04_annotation_guide.md) | Event coding, variable coding, boxes, landmarks, blind re-code | Thu |
| [05_data_dictionary.md](05_data_dictionary.md) | Every table, column, unit | Thu |
| [06_pipeline_architecture.md](06_pipeline_architecture.md) | Stage graph, contracts, provenance, retrieval | Thu |
| [07_evaluation_plan.md](07_evaluation_plan.md) | Every metric, at every layer | Fri |
| [08_statistical_analysis_plan.md](08_statistical_analysis_plan.md) | Pre-specified analysis. Write before seeing results | Fri |
| [09_results_template.md](09_results_template.md) | The empty tables | Sat |
| [10_failure_taxonomy.md](10_failure_taxonomy.md) | Fixed causal categories for the error audit | Sat |
| [11_human_factors_notes.md](11_human_factors_notes.md) | Analyst-in-the-loop sub-study | Sat |
| [12_reproducibility.md](12_reproducibility.md) | Environment, seeds, run identity | Sun |
| [13_interview_talking_points.md](13_interview_talking_points.md) | The demo, and the questions to expect | Sun |
| [14_build_schedule.md](14_build_schedule.md) | Scope tiers and the cut order | Wed |

## The one-paragraph version

McColgan et al. (2026) defined and validated a coding scheme for kickout
strategy in Ladies Gaelic Football, applied it by hand to 2172 kickouts,
and had to discard 908 more — 29.4% — because broadcast footage did not
show them. Their conclusion calls for a central video and tracking
platform to remove that constraint. This project asks how much of their
scheme can be produced automatically from the same broadcast footage,
measures the agreement variable by variable against manual coding, and
reports the coverage rate as the direct counterpart to their exclusion
rate.
