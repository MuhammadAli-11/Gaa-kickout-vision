# 09 — Failure taxonomy

The categories are fixed **before** the audit so that classification is
coding rather than storytelling. Every FP and FN gets exactly one primary
cause and optionally one secondary.

---

## The categories

### 1. `aerial_occlusion`
Bodies overlap at the point of contest. The detector merges two players
into one box, or drops boxes entirely, at precisely the frame that
defines the event.

*Signature:* `n_tracks` dips at `t_peak`; ID switches cluster within
±1 s of the contest; boxes visibly merge in the overlay.

*Why it is structural:* a single camera at pitch level cannot resolve
overlapping bodies. More training data reduces it; it does not remove it.

### 2. `camera_pan_cut`
The director follows the ball or cuts to a different angle, breaking
tracks at the moment of interest.

*Signature:* `scene_change_flag` fires inside the event window;
fragmentation count spikes; track IDs turn over almost completely.

*Why it is structural:* broadcast direction is optimised for viewing, not
for analysis. The camera is doing its job correctly and the analysis is
collateral damage — a good line to have ready, because it reframes the
failure as a property of the data source rather than of the method.

### 3. `scale_distance`
Players on the far side fall below reliable detection size.

*Signature:* recall in the small bbox-area tercile is far below the large
tercile; missed players concentrate in the upper third of the frame.

*Why it is structural:* a GAA pitch is 130–145 m long and 80–90 m wide,
larger than a soccer pitch. At 720p a far-side player may be 15–25 px
tall. This is a sensor-resolution limit, not a model limit.

### 4. `no_ball_ambiguity`
Without ball tracking, the contest is inferred from player behaviour, so
*any* convergence of players resembles a kickout: a melee, a substitution
huddle, a scoring celebration, a free-kick wall, a throw-in.

*Signature:* the false positive looks exactly like a contest in the
feature space; `centroid_spread` compresses and vertical velocity spikes,
but no kickout occurred.

*Why it is structural:* the information required to disambiguate is not
present in the features by construction. This is the honest cost of the
no-ball design decision, and naming it as a cost rather than defending it
is the stronger position.

### 5. `registration_failure`
The shot could not be registered to the pitch, or the held-out
reprojection error exceeded the threshold, so no pitch-space variable
could be computed.

*Signature:* shot absent from `homographies.json`, or `rmse_holdout_m`
above `pitch.max_acceptable_rmse_m`.

*Why it is structural:* a tight shot showing eight players and no pitch
line contains no information to register against. The director framed for
drama, and the geometry went with it.

### 6. `team_assignment_failure`
Players detected and registered, but too many left unassigned for the
defender count to be meaningful.

*Signature:* high `count_uncertainty` on the kickout; low mean confidence
in `track_teams.csv`.

*Why it is structural:* torso colour at 20 px of player height under
floodlights is close to no signal at all. Kit clashes make it worse.

### 7. `annotation_error`
On review, the model was right and the ground truth was wrong.

**Log these. Do not silently fix them.** The count is a direct empirical
estimate of label noise and it belongs in the results beside the
intra-rater κ. Correcting the labels and re-running without saying so
inflates every metric and is undetectable to a reader — which is exactly
why disclosing it is worth so much.

### 8. `other`
Anything unclassifiable. If `other` exceeds ~15% of cases, the taxonomy
is wrong and needs a category added; say so rather than forcing cases
into ill-fitting bins.

---

## Procedure

1. `python src/s09_failure_audit.py --video-id <id>` — exports a padded
   clip per FP/FN and a pre-filled CSV.
2. Watch each clip. Fill `cause` and, where relevant, `secondary_cause`.
   One line of `reviewer_note` per case, written while watching.
3. Re-run `python src/s11_report.py` to regenerate the cross-tab and
   `fig5_failure_causes.png`.

**Classify blind to the error type where you can.** Knowing a case is a
false positive nudges you toward `no_ball_ambiguity`; the clip filenames
make full blinding impractical, but be aware of the pull.

### Boundary errors are separated out
A prediction that overlaps a real event but falls below the tIoU
threshold is recorded as `boundary`, not lumped with true false
positives. It represents a localisation problem, not a detection problem,
and the fix is different. Merging the two hides the real error budget.

---

## The output table

The cross-tab of cause × error type is the most quotable table in the
report. Expected shape (illustrative only — do not pre-fill):

| Cause | Uncodeable | Wrong count | Wrong strategy |
|---|---|---|---|
| aerial_occlusion | | | |
| camera_pan_cut | | | |
| scale_distance | | | |
| no_ball_ambiguity | | | |
| registration_failure | | | |
| team_assignment_failure | | | |
| annotation_error | | | |
| other | | | |

**Cross-reference every uncodeable row against the source paper.**
McColgan et al. excluded 29.4% of their kickouts and attributed it to
camera angles and score replays. Your `camera_pan_cut` and replay rows
are the same phenomenon, arrived at independently and measured
automatically. Saying so — with both numbers side by side — is the
strongest sentence in the write-up.

Each row maps to a design implication, which is what makes the audit an
argument rather than a list:

| Cause | Implication for a real research setup |
|---|---|
| aerial_occlusion | Multiple synchronised viewpoints |
| camera_pan_cut | **Fixed** rigs, not broadcast feeds — precisely the platform the source paper's conclusion calls for |
| scale_distance | Higher resolution, or multiple cameras covering pitch zones |
| no_ball_ambiguity | Ball tracking, or event context from a data feed |
| registration_failure | Fixed cameras with a one-off calibration, instead of per-shot re-registration |
| team_assignment_failure | Higher resolution; or a team roster and kit reference per fixture held in the platform's metadata |
| annotation_error | Multiple coders and formal inter-rater reliability, as the source paper did |

That right-hand column is the transition from "personal project" to "PhD
proposal", and it is the reason the failure audit is worth more than a
better F1.
