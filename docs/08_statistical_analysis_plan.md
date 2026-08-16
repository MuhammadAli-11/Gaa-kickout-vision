# 07 — Statistical analysis plan

**Written before results are inspected.** Date written: ______________

The value of a SAP is that it is falsifiable: if the numbers come out
badly and the analysis is unchanged, the result is credible. If you find
yourself editing this document after seeing a metric, that edit is itself
a finding — log it in §7 rather than quietly making it.

---

## 1. Design

A single-coder feasibility study comparing automated coding of kickout
contests against manual coding on the same footage. Observational, no
intervention, no comparison group. **This is a feasibility study, not a
validation study**, and the report says so in the abstract.

## 2. Units and independence

| Analysis | Unit | Independent? |
|---|---|---|
| Detection metrics | Bounding box | **No** — boxes within a frame share a scene |
| Tracking metrics | Trajectory | No — trajectories share a camera |
| Event metrics | Kickout contest | Approximately yes |
| Agreement | Matched event pair | Approximately yes |

Consequences, applied throughout: the bootstrap resamples **events**, not
frames; and detection intervals are described as *within-clip* precision,
not as evidence about GAA broadcast footage in general. Frames within a
clip are about as independent as repeated measures on one participant, and
treating them as independent observations would understate every interval
by a large factor.

## 3. Estimands

The quantities being estimated, stated before they are computed:

1. Probability that a kickout contest visible in the clip is detected
   (recall at tIoU 0.5).
2. Probability that a detection corresponds to a real contest (precision).
3. Expected absolute error in locating the contest peak, in seconds.
4. Agreement between automated and manual duration coding (ICC, LoA).
5. **The ceiling:** agreement between the coder and their own blind re-code.

## 4. Analyses

### 4.1 Proportions
Point estimate plus 95% **Wilson score** interval. Wald is not used: with
n ≈ 15 and proportions near 1 it produces intervals extending past 1.0.

### 4.2 F1 and AP
F1 point estimate from the confusion counts; interval by **percentile
bootstrap over events**, 10 000 resamples, seed fixed in `config.yaml`.
A conservative interval propagated from the Wilson P/R corners is
reported alongside; where the two disagree materially, both are shown
rather than the narrower one.

### 4.3 Presence agreement — Cohen's κ
Binary presence per 5 s bin, model vs coder. Reported **with** observed
agreement (p₀), expected agreement (pₑ) and prevalence, never alone.

κ is prevalence-sensitive, and kickout-present bins are a minority of all
bins. A high κ under low prevalence and a low κ under low prevalence mean
different things, and the reader cannot tell which they are looking at
without pₑ. **Bin width is fixed at 5 s in advance**; reporting the width
that maximised κ would be a form of p-hacking.

### 4.4 Continuous agreement — ICC(2,1)
Two-way random effects, absolute agreement, single measure.

- *Two-way random*: both "raters" (model, human) assess every event and
  both are of interest.
- *Absolute agreement*: systematic offset must be penalised. Pearson r
  would be 1.0 for a model reliably 2 s long — `test_icc_systematic_offset_penalised`
  asserts exactly this contrast.
- *Single measure*: one automated pass is what would be deployed.

CI by the exact F-distribution method (McGraw & Wong 1996). The
implementation is cross-checked against `pingouin`'s ICC(A,1) to 1e-6 in
`tests/test_statsx.py`.

Descriptors (Koo & Li 2016): < 0.50 poor, 0.50–0.75 moderate, 0.75–0.90
good, > 0.90 excellent. **With n ≈ 12–15 the CI will span two or three of
these bands**, so the interval is reported first and the label second, if
at all.

### 4.5 Bland–Altman
Bias with 95% CI, and 95% limits of agreement with their own CIs (Bland &
Altman 1986, 1999). Difference is defined as **model − human**, stated in
every caption.

Before reporting LoA, difference is regressed on mean to test for
proportional bias. If p < 0.05 for the slope, a single LoA is not a valid
summary and the regression-based limits are reported instead. `s08` warns
automatically when this fires.

Normality of differences is checked visually; with n ≈ 12 a formal
normality test has almost no power and is not run.

### 4.6 The ceiling comparison
Intra-rater κ and ICC from pass 1 vs blind pass 2 on a random 25% of the
timeline. Reported adjacent to the corresponding model metric in the same
table. The ratio (model ÷ ceiling) is given as a summary only, always with
both raw values visible.

## 5. Sample size — stated, not apologised for

No power calculation was performed because the sample is determined by the
footage, not chosen. The consequence is stated explicitly instead:

> With n ≈ 15 events, a 95% Wilson interval on a recall of 0.80 runs
> approximately 0.55–0.93. Differences smaller than roughly 0.25 in
> proportion are not detectable at this sample size. No claim in this
> report depends on distinguishing effects smaller than that.

Rough guides for the write-up: a κ estimate with a ±0.15 interval needs
on the order of 100+ bins; an ICC with a ±0.15 interval needs on the order
of 50+ events. Both are well beyond a 10-minute clip. **That gap is a
result** — it quantifies the corpus a real study would need, which is a
directly useful input to a PhD proposal.

## 6. Multiplicity

Many metrics are reported and **no null-hypothesis significance testing is
performed on the primary outcomes**; everything is estimation with
intervals. No correction is therefore required. The only p-value in the
report is the proportional-bias slope in §4.5, used as a diagnostic for
choosing a summary rather than as a finding.

## 7. Deviations log

Any departure from this plan, with date and reason. An empty table is
suspicious; two or three honest entries are credible.

| Date | Planned | Actually done | Reason |
|---|---|---|---|
| | | | |

## 8. Software

Python 3.10+; `numpy`, `scipy`, `pandas`; `pingouin` for cross-validation
of ICC; `motmetrics` for tracking metrics. Core statistics are implemented
in `src/lib/statsx.py` and unit-tested against known-answer cases, so
every number in the report can be explained at a whiteboard rather than
attributed to a library call.
