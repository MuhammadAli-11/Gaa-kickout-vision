# 12 — Interview talking points

## The 12-minute demo

Assume you get 10–15 minutes and that a supervisor will interrupt. Order
matters: lead with the finding, not the pipeline.

| Min | Show | Say |
|---|---|---|
| 0–1 | The paper on screen | *McColgan et al. threw away 29.4% of their kickouts because the broadcast didn't show them, and their conclusion asks for a central video and tracking platform. I built a prototype of the vision layer for that platform, targeting their primary defensive variable, and measured how much of their coding scheme survives automation.* |
| 1–3 | `overlay.mp4` — a success, then **the labelled failure** | "This one it gets right. This one is a false positive — a substitution huddle. Without ball tracking, any convergence of players looks like a contest." |
| 3–5 | R5 event table + PR curve | F1 **with the interval and the n, every time**. "On fifteen events the interval is this wide. This is a feasibility study, not a validation study." |
| 5–7 | R7 agreement table | The ceiling. "My own blind re-code agrees with itself at κ = X. The model reaches Y. The label quality is the binding constraint before the model is." |
| 7–9 | R3 recall by size + R8 failure cross-tab | "Recall on far-side players is X versus Y for near players. That is not a model problem, it is a sensor problem — and it is the argument for fixed multi-camera capture." |
| 9–11 | `docs/05` stage graph + `query.py` | Ingestion, schemas, provenance, SQL over parquet. "Rights status and content hash are columns from ingest, so access policy and traceability are architectural rather than bolted on." |
| 11–12 | R10 human factors | "I also measured whether the tool actually saved me time, and whether I accepted its mistakes. n=1, so it is a demonstration — but it is the measurement I would scale first." |

## Questions to expect, and honest answers

**"Why did you pick that variable?"**
It is the strongest predictor in their mixed model, it needs no
judgement so ground truth for it is reliable, and it is a counting task in
a pitch region — which is exactly the shape of problem detection plus a
homography can solve. Picking the variable their model says matters most,
rather than the one easiest to demo, is the whole argument.

**"Your agreement is well below their kappa of 0.98."**
It should be, and quoting their figure as my target would mean I hadn't
read the method. Their coders classified kickouts a human had already
found and tagged in NacSport. My pipeline has to locate the kickout,
register the pitch, detect the players and assign them to teams before it
can classify anything, and each of those contributes error their number
never absorbed. Their kappa is the right benchmark for the classification
step in isolation — here is that number separately — and the wrong one
for end to end.

**"How accurate is your pitch registration?"**
Held-out reprojection error, median X metres across registered shots,
never the fit error. It sets the floor: a player within X metres of the
65 cannot be classified reliably, and I report how many players per
kickout fall inside that band. I also perturbed each homography by its own
error and recounted, which gives an empirical uncertainty on the count.

**"Why not just track the ball?"**
Deliberately out of scope. The ball is a few pixels, frequently occluded,
and moves faster than the shutter; ball tracking in broadcast GAA is its
own research problem. Excluding it makes the failure mode explicit and
measurable rather than hidden — `no_ball_ambiguity` is the largest
category in my failure audit, and it quantifies exactly what ball tracking
would buy.

**"Your F1 isn't very good."**
Correct, and the interval is wide. Two things bound it: my own coding
agrees with itself at κ = X, so that is the ceiling; and detection recall
on far-side players is Y, so the features are computed on an incomplete
scene. I would fix the labels and the camera before touching the model.

**"How do I know you didn't tune on the test set?"**
There is no learned component in the headline result — the rule detector's
thresholds are percentiles of the feature distribution, not fitted to
labels. The optional learned mode uses temporal block CV with a purge gap,
because adjacent 3 s windows share 83% of their frames and random k-fold
leaks. `src/lib/cv_blocks.py`.

**"What would you do with a year and a real budget?"**
Fixed multi-camera capture at one venue; a corpus of 200+ contests across
conditions; two independent coders with formal inter-rater reliability;
ball tracking or an event feed for disambiguation; and a proper
analyst-in-the-loop study with 8–12 participants under time pressure.
The failure audit is essentially the specification for that.

**"How does this scale to a season?"**
X GPU-seconds per video-minute, so a 70-minute match is Y and a 30-match
season is Z, embarrassingly parallel because no stage crosses videos. The
storage and catalogue changes are in `docs/05`.

**"Is this legal / GDPR-compliant?"**
Footage stayed local, nothing was redistributed, derived artefacts contain
box coordinates and arbitrary track indices only — no identities, no
biometric templates, no cross-clip re-identification. `rights_status` is a
column in the registry from ingest. Publication would go through TU
Dublin's research ethics process.

**"What went wrong that you didn't expect?"**
Have a real answer. If nothing surprised you, you did not look hard
enough. Candidates: the fixed-window duration artefact that made ICC
structurally zero; a timecode off-by-one; how much of the error budget
turned out to be camera cuts rather than detection.

## The things not to do

- Do not show only successes. A reel of wins tells a supervisor nothing
  about whether you can evaluate your own work.
- Do not quote a metric without n and an interval. Once you do it, every
  subsequent number is suspect.
- Do not claim it works. Claim you measured it.
- Do not hide the intra-rater κ if it is low. It is the most sophisticated
  thing in the project.
- Do not oversell the human-factors piece. n = 1 is a demonstration.

## Send-ahead package

If you get to send anything in advance: the repo link, the 2-page report,
and `overlay.mp4` — in that order of importance. If you can, send the
paper's DOI too, with one line: *this is the coding scheme I automated.* The report should be
readable in four minutes and its **Limitations section should be the
longest one**. That inversion is the signal.
