# Contribution-suite methods

## Separate the evidence

The established whole-record-whitening submission remains 96.8889% mean repeated-CV accuracy. The suite first checks all 100 saved original, whole-record-AR and initial-rest-AR fold scores against identical repeated stratified splits (10 folds × 10 repeats, seed 0). This is reproduction from historical feature arrays; it does not rewrite historical classifiers or rerun their source scripts. Hashes protect recording, original source scripts, submission code and historical arrays.

All new results are exploratory because this same 90-trial recording has already been examined. A blocked CV model can train on both earlier and later trials, separated by a one-trial purge. Only the expanding-window protocol is past-only. Neither protocol evaluates a new participant or recording.

## Central whitening follow-up

The primary question is whether whitening improves decoding when the remaining pipeline and held-out assignments match. The fixed primary comparison uses 60 feature channels and a 2.25-second deadline. Other deadlines are supporting sensitivity analyses. Cue/glove values never enter decoder features.

The exact source reproduction uses unchanged definitions from `ECoG_G25_code/3_oct5_analyses/common.py`, `4_partner_checks/partner_checks.py`, and `4_partner_checks/whitening_audit.py`. Only the relevant definitions and whitening experiments are run; unrelated predictive-movement analyses and external-path scripts are not executed. Source hashes and predictions are exported. Source features preserve eight quarter-second bins, bin-major order and float32 high-gamma power. The contribution pipeline uses its existing feature implementation; cross-pipeline scores are separately labelled. Within each on/off comparison, only whitening changes.

G25 random CV uses 20 independently seeded stratified ten-fold runs (seeds 0–19), not the historical ten-repeat splitter. A trial's correctness fraction is averaged across those seeds. The quoted 11 helped / 0 hurt means increased/decreased fractions across 90 distinct trials, not 11 single held-out mistakes repaired. The source one-sided sign-flip uses 20,000 draws and RNG seed 41; the zero-phase comparison consumes draws before the causal comparison. Reproduction gives 94.3889% → 97.3333%, p=0.0004499775. Serial dependence and prior analysis limit inference. This is not a two-sided McNemar test.

The source AR(10) coefficients are fitted to the whole cleaned recording. Causal filter application is therefore retrospective calibration with held-out signal values. Its first-half filter ends at `onsets[44] + 4*fs`, 217.08 s; train trials 1–45 and test 46–90 gives 39/45 → 40/45. Cue 46 starts at 217.96 s. The source expanding first-half check includes test trials 31–45 before coefficients would be available and is explicitly labelled partly retrospective. Zero-phase filtering remains offline throughout.

The contribution ablation fits AR only on the cue-free pre-task interval and compares matched whitening on/off on seven deadlines under identical random and purged-block splits. Baseline confidence temperatures are independently fitted on inner out-of-fold scores. Paired uncertainty averages repeats inside each trial, then resamples 2,000 five-trial moving blocks. It describes this recording under a short-range serial-dependence assumption. Exported first-repeat exact two-sided McNemar summaries are descriptive and do not replace the source test.

A separate past-only calibration check uses the exact recording midpoint, 211.26 s. Classifier training includes only completed trials before that boundary, with an adjacent-trial purge: 43 training trials, 46 test trials (45–90). The source 45/45 and contribution 43/46 cohorts differ; compare methods only within a protocol. Midpoint-fitted filters cannot justify live predictions before that calibration boundary. Fixed one-second and 2.25-second comparisons, plus later expanding-window predictions, are exported. One-second midpoint holdout gives 40/46 without versus 46/46 with either pre-task or first-half AR; 2.25 seconds gives 40/46 versus 44/46. Perfect accuracy on these 46 explored trials is not proof of generalization.

Mechanism controls reproduce causal source CV for AR orders 5, 10 and 20, and a fitted-free first difference. First difference ties AR(10) at 97.3%, so fitted AR coefficients are not uniquely necessary for the improvement. Later-half pre-/post-whitening spectra illustrate frequency weighting; they are descriptive spectra, not anatomical or causal mechanisms.

Still-hand versus cued Paper separation was not established in the user's follow-up and is withdrawn from the headline. Supplied control scripts lack a numeric replication artifact here; no score is fabricated. This failure neither proves absence of neural task activity nor establishes movement-independent intention. History analyses below are secondary diagnostics.

## Causal predictor

At 1200 Hz, reference each sample over the acquired ECoG channels, high-pass at 1 Hz, and notch 50 Hz harmonics through 300 Hz. Each filter is forward-only with retained state. Fit AR(10) on cue-free pre-task data from 2 seconds through two seconds before the first cue (10.08 seconds), whiten causally, then band-pass 50–300 Hz. Fit neither referencing nor AR to test-period labels or glove values. Chunk size is 120 samples for cleaning/band filters; per-channel whitening uses larger 12,000-sample chunks with preserved FIR state. Prefix and chunk-equivalence checks also run on the actual recording.

The first feature bin starts 0.25 seconds after cue. Each feature is log10 mean square within a 0.25-second bin. Deadlines 0.50, 0.75, 1.00, 1.25, 1.50, 2.00 and 2.25 seconds yield 1, 2, 3, 4, 5, 7 and 8 bins respectively. The deadline is sample waiting time, not compute time or glove reaction time. A one-second model has 0.75 seconds of binned input. Models use shrinkage LDA. ANOVA scores averaged across bins rank channels using training labels only, with stable channel-number tie-breaking.

Every grid setting has identical repeated random splits and five contiguous chronological test blocks with adjacent-trial purging. Nested joint selection compares all 28 settings in three purged inner training blocks, maximizes held-out inner correct count, then breaks ties by earlier deadline and fewer channels. Outer-test labels only score predictions. Outer folds and chosen settings are exported.

For confidence, each fixed 60-channel deadline gets its own temperature per outer fold, fitted on inner out-of-fold LDA scores by minimizing log loss. Outer-test probabilities determine calibration, multiclass Brier score (sum across classes, mean across trials), ten-bin confidence ECE and fixed-threshold coverage curves. A predeclared 0.90 policy accepts the earliest confident prediction, waits to the next deadline otherwise, and returns code 0 (abstain) after 2.25 seconds. Abstentions have no decision timestamp and are explicitly counted. Accepted-only accuracy must be read beside coverage.

Expanding windows train trials 1–44, 1–59 and 1–74, leave an adjacent trial unused, and test 46–60, 61–75 and 76–90. Only 45 trials are tested in this protocol.

## Secondary brain–hand diagnostics

A trial row records cue/gesture, all five starting finger values, their mean starting posture, movement amount, estimated onset, peak speed, previous gesture/amount, rest duration, pre-cue beta and neural outcome power. The first trial has no predecessor, leaving 89 transitions. Current movement amount is the 95th percentile of RMS finger displacement relative to the pre-cue median posture during the two-second cue.

Onset is the first displacement above max(0.02, baseline median + 3 × 1.4826 MAD) sustained for 60 samples (50 ms). Baseline is the second before cue; sensitivity columns use two and four robust deviations. Undetected onset stays missing and never becomes zero. Peak speed uses a trailing 50 ms glove mean. These are estimates from glove values, not measured muscle onset. Movement alignment excludes only undefined alignments; it does not exclude those trials from decoding or explanatory power analyses.

Scientific traces use causal, unwhitened 13–30 Hz beta, 50–300 Hz high-gamma and 1–30 Hz low-frequency voltage. Power displays use trailing 50 ms mean square, then dB. Glove and neural traces are saved at 200 Hz for replay. Power outcomes average channel log power from 0.5–2.0 seconds. Low-frequency traces are event-related voltage displays, not a clinical ERP diagnosis.

The small base OLS model controls two gesture contrasts, mean starting posture, current movement amount, trial number and rest duration. The history extension adds only preceding movement amount. Peak speed and onset are recorded for diagnostics but omitted from the small explanatory model to limit degrees of freedom. A secondary pre-cue-beta association is labelled separately and is not a new decoder claim.

Compare models on five purged chronological blocks with scaling fitted only on training rows. Report held-out MSE/R² and an adjusted coefficient in dB per full-recording SD of preceding movement. The descriptive coefficient uses all 89 transitions; 1,000 moving-block bootstrap resamples of five consecutive trials produce a within-recording 95% interval. This assumes short-range serial dependence and is not participant-population uncertainty. Current movement can mediate history as well as confound it; adjustment changes the question and cannot establish a causal history effect.

Repeat absolute power, common pre-task subtraction, and each trial's own pre-cue subtraction. A common reference is a constant shift, so the intercept makes its effects identical to absolute power. Trial baseline changes can reveal baseline overlap or other dependence; they do not identify its mechanism.

The [2025 MEG study](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1547916/full) motivates inspecting whether a 2–3-second rest is a clean baseline. Its approximately 4–5-second post-movement beta return in a button-press task does not establish that duration here. Neurophysiology, stimulus timing, movement execution and starting state are not fully separable in this recording.

## Spatial, acquisition and quality

Numbered layout is 10 rows × 6 columns: electrode `n` maps to row `(n−1) mod 10`, column `floor((n−1)/10)`, as documented by the supplied project material. Local referencing subtracts available orthogonal grid neighbors. Neighbor relations are layout-based, not anatomical distances. Selection frequency comes from repeated training folds. Redundancy is mean feature correlation over outer training folds. Contributions exactly decompose held-out Scissors-minus-Rock LDA score after centering on the training mean; Paper's score is still needed for classification.

Feature budgets retain all 60 channels for preprocessing. Acquisition subsets select 10/20/40 channels inside each outer training set and then recompute CAR, pre-task AR, filtering and classification using only those raw rows. Their calibration OOF predictions also reselect channels and recompute preprocessing within each inner training block. Persistent loss experiments remove 1/3/6 channels with five deterministic seeded patterns, preprocess the altered montage and retrain every fold. These measure robustness after retraining, not frozen-model sudden-dropout tolerance.

Quality records flat pre-task channels, nonfinite samples, 50-Hz PSD ratio to neighboring bands, and large sample differences relative to pre-task median/MAD. Trial transient threshold is 12. Flags remain visible without exclusion or relabelling. Nonfinite input and unusable flat calibration abort with a clear error rather than silently repairing data. Quality diagnostics do not indicate disease or seizure activity.

Cleaning chunk p50/p95 and classifier inference p50 are measured separately from recording sample waiting and glove onset. Pipeline wall time includes offline feature preparation. Replay is recorded-data playback, not proof of deployed live end-to-end latency.
