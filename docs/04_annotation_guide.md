# 03 — Annotation guide (SOP)

Five annotation jobs. Priority order matters more than usual here,
because pitch landmarks are on the critical path — nothing in pitch space
exists without them.

| Job | Output | Effort | Priority |
|---|---|---|---|
| A. Event coding | `gt_events.csv` pass 1 | 60–90 min | **Essential** |
| B. Variable coding | `gt_coding.csv` pass 1 | 60 min | **Essential** — this is what you validate against |
| C. Blind re-code | both files, pass 2 | 30 min | **Essential** — it is the ceiling |
| D. Pitch landmarks | `pitch_keypoints.csv` | 60–90 min | **Essential** — blocks everything in metres |
| E. Box annotation | `gt_boxes.txt` (MOT16) | 3–5 h | Important, cuttable to 300 frames |

Do B using the definitions copied from the paper's Table 1
([docs/03](03_operational_definitions.md)), not from memory of them.

---

## A. Event coding

**Tool:** any player with frame stepping. VLC (`E` steps one frame),
Kinovea, or the browser. Record frame numbers, convert to seconds by
dividing by 25 — or read `timestamp_s` straight from
`data/interim/<video_id>/manifest.csv`.

**Procedure**

1. Watch the full clip once, no coding, just to build expectation.
2. Second pass: for each kickout, step frame by frame to find `t_start_s`,
   then `t_peak_s`, then `t_end_s` per [docs/02](02_operational_definitions.md).
3. Record `n_players_in_contest` at `t_peak_s`.
4. Set `coder_id` to your initials and `coding_pass` to `1`.
5. Log anything ambiguous in `annotation_notes.md` *at the time*.

**Do not** look at any model output before finishing pass 1. If you have
already seen predictions, your labels are contaminated and the agreement
statistics are inflated in a way that cannot be undone.

Template: `data/gt/_templates/gt_events_template.csv`.

---

## B. Variable coding — the paper's scheme

One row per kickout in `gt_coding.csv`. For each, using their Table 1
definitions and the pitch markings visible in frame:

1. **`n_defenders_in_65`** — count the defending team's players inside
   the opposition 65 m line during the 1.5 s before the kick. Use the
   same window the pipeline uses, or the comparison is unfair. If the
   camera does not show the whole press, code `unclear` and move on —
   do not estimate.
2. **`n_defenders_band`** — derived from the count, but record it
   explicitly so a band error can be told apart from a count error.
3. **`defensive_strategy`** — zonal, player-to-player, or concede, per
   their definitions.
4. **`distance_class`** — short or long, split at the 45.
5. **`contested_proxy`** — their contested definition, opponent within
   2 m of the player gaining possession.
6. **`kickout_won`** — record it even though nothing models it. It costs
   nothing and lets you say which outcomes your uncodeable kickouts had.
7. **`confidence`** — high, medium, or low, for *your own* coding.

Point 7 matters more than it looks. When the model disagrees with a
low-confidence label, that is not straightforwardly a model error, and
being able to split the agreement statistics by label confidence is a
genuinely sophisticated piece of analysis that costs one extra column.

---

## D. Pitch landmark annotation

Run `python src/s04_register_pitch.py --video-id <id> --list-shots` to
get the shot list, then annotate **only the shots containing a kickout**.
Registering every shot is usually wasted work.

Per shot, on one keyframe:

- Click **6–8 landmarks**, well spread across the frame. Four is the
  minimum for a homography and leaves nothing for validation.
- Prefer intersections of a cross-pitch line with a touchline — they are
  unambiguous. Rectangle corners help on tighter shots.
- **Spread them.** Eight landmarks clustered in one corner give a
  homography that is superb there and useless at the far 45, and a
  reprojection error that hides the problem completely.
- Use consistent naming. Inconsistent landmark names across shots is the
  single most common way to waste an hour of this work.

Then `--fit` and check the **held-out** RMSE. Under 2 m is workable;
over that, either add landmarks or exclude the shot. Do not code
kickouts from a shot whose registration you do not trust — an
unregistered shot belongs in the coverage denominator, which is a
result, not in the results as a guess.

**Camera motion within a shot.** One homography per shot is an
approximation, because broadcast cameras pan and zoom continuously. Check
it: register the same shot from two keyframes, 3 s apart, and compare the
projected position of a static landmark. That drift, in metres, is the
approximation error, and it belongs in the limitations.

---

## E. Box annotation for detection and tracking metrics

**Subset:** 300–500 frames. Not random — sample **three contiguous
30–60 s blocks** that each contain a kickout, plus one block of open
play. Contiguity is required because tracking metrics (IDF1, ID switches)
are meaningless on non-adjacent frames.

**RECORD** which blocks you chose and why. Choosing only clean wide shots
inflates every number; you want at least one block containing a camera cut.

**Tool:** CVAT (recommended — has interpolation and MOT export) or Label
Studio. Single class, `player`.

**Rules**

- Box every person on the field of play, including officials — then use
  the `cls` column so officials can be excluded at evaluation time. Do not
  simply skip them; an unboxed referee becomes a false positive that is
  really an annotation gap.
- Exclude: crowd, bench, camera operators, anyone off the field.
- Occluded players: box the **visible extent**, set `visibility` to the
  approximate visible fraction (CVAT writes this into MOT format). The
  occlusion column is what lets you show detection failing *specifically*
  at the aerial contest.
- Truncated at frame edge: box the visible part.
- Track IDs must persist through occlusion. If you genuinely cannot tell
  which player emerged from a ruck, start a new ID and note it — your own
  uncertainty here is data about how hard the task is.

**Export:** MOT16 `gt.txt`, one line per box:
`frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility`,
frames 1-based (the loader in `s06` converts to 0-based).

**Time management.** This is the slowest task in the project. Start it
early, and if it is not finished by Saturday morning, cut it to 300 frames
rather than cutting the evaluation. Detection metrics on 300 well-chosen
frames are worth more than metrics on 500 rushed ones.

---

## C. The blind re-code — the most important 30 minutes

Wait **at least 48 hours** after pass 1. Then:

1. Randomly select 25% of the timeline — not 25% of events; select time
   blocks, so that a kickout you missed entirely in pass 1 can show up in
   pass 2. Use `python tools/sample_recode_blocks.py` (or a die).
2. Re-code those blocks **without looking at pass 1** — both the event
   boundaries *and* the paper's variables. Cover it, close it, put it in
   another directory.
3. Append rows with `coding_pass = 2` and the same `event_id` where the
   event corresponds to a pass-1 event; use a fresh `event_id` where it does not.

`s08_agreement.py` computes intra-rater kappa on presence and ICC(2,1) on
duration from these rows. **That number is the ceiling on any model score
in this project.** No automated system can agree with your coding better
than your coding agrees with itself.

Reporting the model's F1 next to your own self-agreement is the single
most researcher-like move available in a four-day project, and almost no
interview candidate does it.

### If your intra-rater agreement is poor

κ below 0.60 is not a disaster to hide — it is a finding about the task.
But before reporting it, check the diagnosis:

- **Boundaries disagree, presence agrees** → the definitions in docs/02
  are too loose. Tighten them, re-code, and report both versions.
- **Presence disagrees** → you are missing events on one pass, usually
  because of camera cuts. This is a genuine limit of broadcast coding and
  belongs in the results.
- **`n_players_in_contest` disagrees badly** → expected. Report it as the
  least reliable variable and explain why (no pitch calibration, so "within
  5 m" is judged by eye from a moving camera).
