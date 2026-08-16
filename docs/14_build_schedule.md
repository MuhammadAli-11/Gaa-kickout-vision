# 14 — Build schedule and scope tiers

## Read this first: the honest scope warning

Anchoring on the paper made the project better and bigger. Pitch
registration and team assignment are both real subsystems, and neither
existed in the pixel-space version. **The full sixteen-stage pipeline is
a two-to-three week build, not a four-day one.**

So the project is tiered. Build Tier 1 completely before touching Tier 2.
A finished Tier 1 with honest validation is a far stronger interview
artefact than a half-built Tier 3 with no numbers in it.

---

## Tier 1 — the minimum defensible deliverable (target: Sunday)

**Claim:** *Their primary defensive variable — the number of players a
team commits inside the opposition 65 — can be counted automatically from
broadcast video, and here is how well, with what uncertainty, and on what
fraction of kickouts.*

| Stage | Needed? | Notes |
|---|---|---|
| s00–s03 ingest, prepare, detect, track | Yes | Pretrained YOLO. Minutes of GPU, not hours |
| s04 register pitch | **Yes** | Hand-click landmarks on the shots containing kickouts ONLY. Roughly 12–20 shots, 8 landmarks each, ~60–90 min |
| s05 features | Yes | Both passes |
| s06 assign teams | **Yes** | Colour clustering. If it fails, hand-assign teams for the kickout frames only and say so |
| s07 detect kickouts | **No — skip** | Use your manually coded timestamps. This isolates *coding* accuracy from *detection* accuracy, which is the cleaner experiment anyway |
| s08 code kickouts | Yes | `--from-gt` |
| s09 evaluate detection | Reduced | 300 annotated frames, not 500 |
| s11 evaluate coding | **Yes — never cut** | Defender count only is enough |
| s12 failure audit | Yes | Every kickout you could not code, classified |
| s13–s14 overlay and report | Yes | Overlay must show the pitch-space projection, not just boxes |

**Tier 1 results:** homography error in metres; defender-count ICC,
Bland–Altman and exact-match rate against your coding; the intra-rater
ceiling; the coverage rate versus their 29.4% exclusion rate; a failure
cross-tab. That is a complete, publishable-shaped feasibility result.

## Tier 2 — if Tier 1 is done by Saturday night

- s07 automatic temporal kickout detection, evaluated at tIoU 0.3/0.5/0.7
- Defensive strategy classification (zonal / player-to-player / concede)
  with a threshold sweep on nearest-opponent distance
- Short versus long classification

## Tier 3 — say it, do not build it

- Offensive strategy classification (Flat 4 is the tractable one: four
  players across the 21)
- Kickout outcome, which needs the ball
- The zone map from their Figure 1
- s15 human-factors sub-study

Tier 3 items belong in "what I'd do next", which is a section a supervisor
reads closely.

---

## Day by day

| Day | Task | GPU |
|---|---|---|
| **Wed night** | Read the paper properly. Copy Table 1 definitions into the coding manual. Source footage. s00, s01. YOLO smoke test on 30 frames | No |
| **Thu** | s02, s03 end to end. s05 `--image-only`. Shot list from s04 `--list-shots` | Optional |
| **Thu night** | Manual coding pass 1: `gt_events.csv` then `gt_coding.csv`, using their definitions | No |
| **Fri morning** | **Pitch landmark clicking.** The long pole in Tier 1. Register only shots containing kickouts | No |
| **Fri afternoon** | s04 `--fit`, check held-out reprojection error is under 2 m. s05 full. s06 teams, review low-confidence tracks | No |
| **Fri night** | s08 `--from-gt`. First look at the defender counts | No |
| **Sat morning** | s09, s11. All statistics, all intervals | No |
| **Sat afternoon** | **Blind re-code pass 2** — 48 h after Thursday night | No |
| **Sat night** | s12 failure audit. Classify every uncodeable kickout | No |
| **Sun morning** | s13 overlay with pitch projection, s14 figures and tables | No |
| **Sun afternoon** | Write the report. Tier 2 only if genuinely spare | No |
| **Sun evening** | README, `make check`, push | No |

## Cut order

1. Tier 2 and Tier 3, entirely
2. Box annotation 500 → 300 frames
3. Number of registered shots — but never below the kickouts you coded
4. Footage length (**and say so**)

**Never cut:** the paper's definitions, the blind re-code, s11, or the
failure audit.

## Traps specific to this build

- **Landmark clicking always overruns.** Register three shots, fit, check
  the error, and only then do the rest. Discovering after two hours that
  your landmark naming was inconsistent is the worst outcome available.
- **The 48-hour gap is real.** Code Thursday night so the blind re-code
  lands Saturday afternoon.
- **Team assignment may simply fail** on a distant wide shot with similar
  kit. Decide the fallback in advance: hand-assign teams at the kickout
  frames, or report *total* players inside the 65 rather than by team,
  and state which you did and why.
- **Do not tune the strategy thresholds against your ground truth and
  then report accuracy on the same kickouts.** If you sweep the
  nearest-opponent threshold, report the whole curve, not the peak.
