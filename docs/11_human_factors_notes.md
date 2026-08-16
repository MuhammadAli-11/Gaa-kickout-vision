# 10 — Human factors sub-study

## Why this is in a computer-vision project

The studentship is titled *Human Factors and Ergonomics in Sport
Performance Intelligence*. Computer vision is the method; the research
question is whether coaches and analysts actually adopt these systems
under match-day pressure. A project that stops at F1 answers half of the
advertised question.

This sub-study costs about two hours and produces three numbers that most
candidates will not have. It is genuinely small — that is the point. It
is not a study; it is a demonstration that you know the difference between
model accuracy and system usefulness.

---

## Protocol

**Participants:** n = 1 (you). Stated as such. This is an instrumented
self-observation, not a user study, and it is described that way in the
write-up. Overclaiming here would undo the credibility the rest of the
project buys.

**Design:** within-subject, two conditions, on **non-overlapping**
passages of footage to avoid learning effects.

| Condition | Task |
|---|---|
| A — `manual_from_scratch` | Code kickouts with no model output visible |
| B — `model_assisted` | Accept / reject / adjust the model's candidates |

**Measures per decision:** model confidence (hidden in the `--blind`
control), verdict, seconds to decide.

```bash
python src/s12_review_tool.py --video-id <id> --condition manual_from_scratch
python src/s12_review_tool.py --video-id <id> --condition model_assisted
```

**Order:** do A first. Doing B first anchors your judgement about where
events are and contaminates A.

---

## The three questions

### 1. Does the tool save time?
Median seconds per event coded, A vs B. Report medians and the range, not
a t-test on n = 1.

The interesting outcome is a **negative** one: if verifying a candidate
takes as long as finding it from scratch, the tool has moved work rather
than removed it, and that is a finding worth stating plainly.

### 2. Is the confidence score trustworthy?
Acceptance rate by confidence band. If acceptance does not rise
monotonically with confidence, **the score should not be shown to a user**
— an uncalibrated confidence number is worse than no number, because it
invites misplaced trust.

### 3. Does the tool make the human worse?
Acceptance rate of model outputs that are actually false positives.
This is **automation bias**: the tendency to accept an automated
suggestion because it is there. A system that raises throughput while
degrading accuracy is a bad system, and measuring the trade-off is exactly
the human-factors contribution the studentship is asking about.

---

## What to say about it

Two or three sentences in the report, and one line in the interview:

> The model's accuracy is only half of the question. I also measured how
> long it took me to verify a candidate versus code from scratch, and
> whether I accepted the model's mistakes when it showed them to me. On
> n=1 that is a demonstration rather than a result, but it is the
> measurement I would scale up first, because a detector that is 10%
> better and gets ignored by analysts is worth less than one that is 10%
> worse and gets used.

---

## Scaling this up (the PhD-shaped version)

Worth having ready if asked "how would you turn this into a study?":

- **Participants:** 8–12 performance analysts, counterbalanced order.
- **Design:** within-subject, three conditions — manual, assisted with
  confidence shown, assisted with confidence hidden.
- **Measures:** time per event, coding accuracy against a consensus
  reference, NASA-TLX for workload, a trust scale, and post-task
  interviews on where they stopped checking the model.
- **Primary outcome:** accuracy per unit time, not accuracy alone.
- **The interesting hypothesis:** showing an uncalibrated confidence score
  *increases* throughput and *decreases* accuracy — i.e. the interface
  choice matters more than the model improvement.
- **Match-day extension:** the same task under time pressure, since the
  advertised research question is specifically about high-pressure
  environments, and lab-condition results may not transfer.
