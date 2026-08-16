# 01 — The source paper, and what this project does with it

## The paper

> McColgan, A.; Bradley, J.; Earle, D.; Gaul, D.; Martin, D. (2026).
> *Identifying and Defining Kickout Strategies in Senior Inter-County
> Ladies Gaelic Football.* **Applied Sciences** 16(7), 3277.
> <https://doi.org/10.3390/app16073277>

**Licence: CC BY 4.0.** You may reuse the text, tables and figures,
including Table 1's operational definitions and the zone map in Figure 1,
provided the original is clearly cited. Reproduce their definitions
verbatim in `data/gt/_templates/coding_manual.md`, cite the DOI beside
them, and code against those exact words rather than a paraphrase — a
paraphrased definition is a *different* definition, and the whole point of
building on this paper is that you are measuring against a scheme that
was expert-validated and published rather than one you invented.

## Why this paper and not another

Four reasons, in descending order of importance.

**1. Their conclusion asks for the thing the studentship is advertising.**
The paper closes by calling for a central Gaelic Games intelligence
platform to standardise protocol, collect video, and potentially hold
tracking data from senior inter-county games — because the research they
want to do is bottlenecked on the video, not on the analysis. The TU
Dublin studentship is for building the video data environment, ingestion
through retrieval, with a computer-vision layer. The paper states the
requirement; the studentship is the response. Building your project in
that gap means you are not proposing a project, you are prototyping the
one they already said they need.

**2. Their central limitation is a computer-vision problem, and it is
measurable.** They analysed 2172 kickouts out of 3081 available. The
other 908 — 29.4% — were unusable because broadcast camera angles and
score replays hid the kickout. Worse, they note the loss is not random:
stronger teams score more, scores trigger replays, replays hide the
following kickout, so the missing data is biased toward the very teams
whose strategies matter most. **That exclusion rate is the single most
useful number available to you.** Your pipeline's own coverage rate
measures the same problem from the other side, and comparing the two is a
result no amount of model tuning could buy.

**3. Their variables are geometry.** The defensive strategies are defined
by where players stand relative to pitch lines and to each other: a zonal
press occupies space inside the opposition 65, a player-to-player press
keeps defenders within about a metre of an opponent, conceding means not
pressing at all. Player detection plus tracking plus a homography
produces exactly those primitives. This is unusually lucky — most
performance-analysis coding schemes rest on judgements a CV pipeline
cannot express, and this one largely does not.

**4. It sits in the supervisor's research lineage.** Dr Collins is a
co-author on work this paper builds on and cites, including the analysis
of kickout effectiveness in sub-elite Gaelic football and the benchmarking
of elite ladies Gaelic football performances. The paper itself was funded
by the TU Dublin PhD Scholarship Programme. You are not arriving with an
outside idea; you are extending a line of work already running in the
group.

## What the project therefore becomes

**Not:** "I detected kickouts in video."

**Instead:** *Can the McColgan et al. (2026) kickout coding scheme —
currently produced by hand in NacSport, at a cost of 29.4% of all
kickouts being discarded — be reproduced automatically from the same
broadcast footage, and how much of it survives the attempt?*

That reframing changes three things:

| | Before | After |
|---|---|---|
| Ground truth | Definitions you invented | A published, expert-panel-validated scheme |
| Reliability benchmark | None | Their inter-operator kappa, 0.981–1.00 on 200 kickouts |
| Output space | Pixels | Pitch metres — because every one of their variables is defined by a pitch line |

The third is the biggest engineering consequence. Their scheme counts
players inside the opposition 65, splits kickouts at the 45, defines
contested as an opponent within 2 m, and defines the Flat 4 by four
players across the 21. **None of that is expressible in pixels.** A
homography is therefore not optional polish, it is load-bearing, which is
why pitch registration is its own stage with its own error metric.

## The variable map — what to attempt, and what not to

| Their variable | Automatable? | This project | Why |
|---|---|---|---|
| **Players inside the opposition 65** | **Yes** | **PRIMARY target** | Counting in a pitch region. No judgement, so ground truth is reliable. Strongest predictor in their model |
| Defensive strategy: zonal / player-to-player / concede | Mostly | Secondary | Their definitions are geometric; the classifier is three rules on nearest-opponent distance and defender count |
| Short vs long kickout | Partly | Secondary | Needs the reception point, which without ball tracking is a cluster-centroid proxy |
| Contested (opponent within 2 m of receiver) | Proxy only | Exploratory | "Player gaining possession" needs the ball. Report as a proxy, under a different name |
| Time taken for kickout | Partly | Stretch | Needs the ball going dead, which is a separate event-detection problem |
| Kickout zones 1–9 | Yes, given registration | Stretch | Falls out of the homography once their zone map is reproduced |
| Kickout won / lost | **No** | Out of scope | Requires the ball and possession |
| Offensive strategies: Bunch and Break, Overload, Flat 4, Purposeful vs Limited Movement | **Mostly no** | Out of scope | Depends on run intent and on a signal between keeper and outfield players. Flat 4 alone is nearly tractable — four players on the 21 — and is the obvious next one |
| Shots, scores, net points per kickout | **No** | Out of scope | Downstream of possession |

Naming the "no" rows with a reason is worth as much as delivering the
"yes" rows. It shows you read the scheme rather than skimmed it.

## The two things to say in the interview

**On the benchmark.** Their reported inter-operator kappa of 0.981–1.00
is not a target you should expect to hit, and claiming otherwise would
signal you had not thought about it. Their coders were agreeing on the
*classification* of kickouts a human had already found and tagged in
NacSport. This pipeline must locate the kickout in time, register the
pitch, detect players, assign them to teams, and only then classify.
Their figure is the right benchmark for the classification step in
isolation and the wrong one for the end-to-end system. Report both
framings and say which is which.

**On the contribution.** Their manual system discards 29.4% of kickouts
and does so non-randomly. A pipeline that codes 60% of kickouts
automatically at moderate accuracy is not obviously better than a human
coding 70% at near-perfect accuracy — but it is a different trade-off,
it scales to whole seasons, and the failure analysis says exactly what
capture setup would remove the constraint. That is the argument, and it
is an argument about data infrastructure, which is what the studentship
is about.

## Two things to check before you start

**Code and cohort.** The paper analyses Ladies Gaelic Football,
2019–2023 TG4 All-Ireland Senior Championship, where the goalkeeper may
kick from the hand. If your footage is men's Gaelic football, the
kickout is taken from a tee, the range is longer, the drop zone the paper
identifies is wider, and their strategy frequencies will not transfer.
That does not invalidate the exercise — you are testing whether their
*coding scheme* can be automated — but state which code your footage is
and do not quote their frequencies as expectations for it.

**Rule changes.** Men's Gaelic football has been through a significant
rule-change process; the paper itself cites the Football Review
Committee. Kickout regulations are among the things that have been under
review, and any change to how far a kickout must travel would directly
alter contest frequency and location. **Verify the current rules for the
season your footage comes from before writing anything about expected
kickout behaviour**, and note the season in the report.
