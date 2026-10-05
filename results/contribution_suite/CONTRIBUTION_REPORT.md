# Whitening Under Test

Fast, reliable ECoG gesture decoding: Team G25 source reproduction and controlled follow-up. Updated 6 October 2026.

## Central contribution
Whitening is the strongest supported preprocessing improvement in this project. We test it with matched predictions, causal filtering, earlier-signal calibration, speed and reliability. It is not claimed to be the uniquely best possible decoder or transform.

Established submission stays **96.9% mean repeated 10-fold × 10 CV**. The old 97.8% was one 88/90 run. New numbers below use distinct protocols and remain exploratory.

## Reproduce the supplied G25 follow-up
The original causal-filter comparison gives **94.4% → 97.3%**, a **2.94 percentage-point** improvement. **11 trials improved** their average correctness fraction across matching CV seeds and **0 worsened**.

Original one-sided Monte Carlo sign-flip p = **0.00044998** (rounded **0.0004**), from 20,000 draws with seed 41 after the zero-phase test consumes the RNG. The test uses 90 per-trial correctness-fraction differences across 20 ten-fold seeds, not 1,800 independent trials. Serial dependence and prior exploration limit inference.

Source: `ECoG_G25_code/4_partner_checks/partner_checks.py`, shared `3_oct5_analyses/common.py`, and order-control definitions from `4_partner_checks/whitening_audit.py`. Source hashes accompany the exported predictions.

The headline filter application is causal, but its AR coefficients are fitted to the whole recording. It is retrospective within-recording evidence. Causal filtering alone does not make its calibration prospective.

## Calibration that ends before test samples
The team's causal first-half check trains the classifier on trials 1–45 and tests 46–90. AR fitting stops at **217.08 s**, before cue 46 at **217.96 s**. The score is **39/45 without whitening → 40/45 with whitening**, or **86.7% → 88.9%**. This is a modest one-trial gain, not 97.3% prospective accuracy.

Our stricter time-midpoint comparison fits AR only before 211.26 s, purges one adjacent classifier-training trial, and tests 46 later cues. The 45/45 and purged 43/46 cohorts are different; compare only paired methods within each protocol.

| Time-midpoint holdout | Method | Correct / tested | Accuracy |
|---|---|---:|---:|
| 1.00 s | off | 40/46 | 86.96% |
| 1.00 s | pre_task | 46/46 | 100.00% |
| 1.00 s | first_half | 46/46 | 100.00% |
| 2.25 s | off | 40/46 | 86.96% |
| 2.25 s | pre_task | 44/46 | 95.65% |
| 2.25 s | first_half | 44/46 | 95.65% |

## Matched pre-task-fit whitening ablation
Fit AR(10) using only cue-free 2.00–10.08 s. CAR, causal high-pass/notches/band filter, channel count, binning, LDA and split indices match across whitening on/off. The fixed primary deadline is 2.25 s; other deadlines are sensitivity/supporting comparisons.

| Fixed 2.25 s / 60 channels | Without | With | Gain | Within-recording block interval |
|---|---:|---:|---:|---|
| random | 95.00% | 97.44% | +2.44 pp | [0.67, 4.67] pp |
| blocked | 92.22% | 97.78% | +5.56 pp | [0.00, 10.00] pp |

These 10-fold × 10-repeat splits differ from the team's 20-seed comparison. For uncertainty, repeat outcomes are averaged inside each original trial, then five-trial blocks are resampled. Exact two-sided McNemar summaries use only one prediction per trial from the first repeat; they are descriptive and do not replace G25's one-sided sign-flip.

## Mechanism control and limits
A fitted-free first-difference filter reaches the same 97.3% in the G25 causal CV scheme. AR(5) and AR(20) are similar. This supports frequency weighting / temporal decorrelation as useful processing, while weakening a claim that fitted AR coefficients are uniquely necessary. Whitening adds no recorded information and does not remove all artifacts.

| Source causal sensitivity | Accuracy |
|---|---:|
| AR(10) | 97.33% |
| first_difference | 97.33% |
| AR(5) whole-record | 97.17% |
| AR(20) whole-record | 97.22% |

## Fast decisions and reliability
The pre-task-whitened one-second model gives **88/90** under purged chronological evaluation. The fixed 0.90 confidence policy accepts **77/90**, correctly classifies **75/77**, and abstains on **13**. Processing time, sample waiting time and glove reaction time remain separate.

Training-only feature selection, referencing, acquired-channel subsets and persistent channel loss after retraining remain supporting checks. Results cannot be transferred to an unseen participant or recording.

## Withdrawn hypothesis and retained diagnostics
The user reports that still hand versus cued Paper did not hold up in follow-up. Reliable separation is not established and is withdrawn from the central story. The provided same-hand/paradox scripts describe controls but do not include a numerical replication output here; no score is invented.

Our earlier movement-history analysis was independently inconclusive after accounting for actual hand behavior. It remains a secondary diagnostic appendix. Neither high gesture accuracy nor an inconclusive rest/Paper test establishes movement-independent intention or absence of a neural task signal.

## Scientific figures
- [06 whitening evidence](figures/06_whitening_evidence.png)
- [07 calibration boundary](figures/07_calibration_boundary.png)
- [08 whitening spectrum](figures/08_whitening_spectrum.png)
- [09 whitening mechanism control](figures/09_whitening_mechanism_control.png)
- [05 speed and confidence](figures/05_speed_and_confidence.png)

Figures 01–04 retain annotated brain/glove and history cases as secondary diagnostics. Prior output artifacts are archived separately, so the change in project direction is traceable.

## Limits and reproduction
- One previously explored participant recording; all new paired comparisons are exploratory.
- Repeated CV has 900 prediction exposures but only 90 distinct trials; its p-value is not computed over 900 independent observations.
- McNemar is descriptive here; serial dependence and prior model exploration limit inferential claims.
- Whitening reshapes predictable temporal structure and frequency weighting; it adds no recorded information and does not remove every artifact.
- Whitening supports requested-gesture decoding, not proof of movement-independent intention or separation of still hand from cued Paper.
- Causal filtering and past-only calibration are separate requirements. First-half calibration cannot justify first-half live predictions.
- This fixed CAR ablation does not establish superiority to every possible decoder or montage.
- One previously explored recording and participant; all new comparisons are exploratory.
- Repeated CV predictions reuse 90 trials; they are not independent participants or new trials.
- Cue and visual stimulus identify the requested gesture; motor and stimulus contributions are not fully separable.
- Glove-defined movement onset is an estimate; no detectable displacement is not proof of no motor activity.
- Calibration and adaptive decisions are evaluated within one recording, not prospective deployment.
- Channel loss is persistent loss with retraining, not unexpected dropout handled by a frozen model.
- Local referencing and channel correlations do not locate exact sources or demonstrate causal connectivity.
- Lower high-gamma power does not establish reduced effort; failed decoding does not establish forgetting.
- Whole-record AR fitting uses held-out signal values: causal filter application is not fully prospective calibration.
- Original first-half-calibrated expanding evaluation includes trials 31–45 before calibration finishes; it is retrospective for those trials.
- The first-45 → last-45 causal half-split has calibration finished before test cue 46, and is the relevant first-half check.
- The source p-value uses a one-sided sign-flip, not two-sided McNemar; time dependence and prior exploration limit inference.

Run `python -m ecog_contribution run`; launch `streamlit run dashboard/app.py`. The dashboard reads cached results and never fits models. CSV/NPZ predictions, splits, calibration coefficients, configuration and source hashes are exported. Original recording and submission files stay unchanged.
