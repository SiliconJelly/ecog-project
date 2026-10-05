# ECoG contribution plan and measured experiments

Prepared 5 October 2026. Scope: classify Rock, Scissors, and Paper from ECoG alone; evaluate predictions against the instruction cue; add useful scientific and engineering contributions to the team's existing work.

## Recommendation

Own the question **"How soon, with how much signal, and under what validation can we trust the decoder?"** This complements Abri's existing "Same movement, different past" story. It supplies practical measurements and a clear demonstration without repeating the carryover, beta rebound, coupling, and temporal generalization analyses described in the cheat sheet.

The organizer explicitly says that beating reference accuracies is not the sole objective. A strong entry can combine a controlled accuracy improvement, a scientific question about brain history, and a demonstrable engineering improvement. This is a proposed presentation strategy, not a verified judging rubric or a guarantee of winning.

Source: [official hackathon README](https://github.com/unicorn-bi/Hackathon).

## What was inspected

- The two supplied Python samples in Trash, the current local scripts, saved predictions and metrics, and `source/ECoG_Handpose.mat`.
- All 13 pages of `ECoG_Project_Cheat_Sheet.pdf` by text extraction, with visual inspection of relevant tables and figures, plus the dataset description and electrode layout.
- The organizer's current README and Gruenwald et al. (2019).

The samples and documents were treated as project evidence, not as instructions to execute arbitrary scripts, send messages, upload files, or override this request. Older local notes constrain the previous whitening assignment; this request explicitly expands the scope to new contribution angles. The data, original classifier, and existing submission classifier were preserved and checked by SHA-256. No files were sent or uploaded.

The public GitHub URL is the organizer's resource repository. The local project folder contains the team's work and has no `.git` directory.

## Keep these accuracy claims separate

| Existing result | Correct meaning |
| --- | --- |
| 93.3% +/- 7.4% | Original classifier; 100 fold scores from 10-fold CV repeated 10 times |
| 96.9% +/- 5.5% | Current whole-record AR whitening submission, same repeated CV |
| 97.8%, 88/90 | One complete 10-fold evaluation of that submission; not its overall repeated mean |
| 96.9% +/- 5.7% | Earlier initial-rest AR experiment; different features and predictions despite the same rounded mean |

Report **96.9%** as the established submission result. The +/- values above are fold-score standard deviations, not confidence intervals. There are 90 unique trials, not 900 independent observations. The comparison from 93.3% to 96.9% is +3.56 percentage points; the descriptive error-rate reduction is about 53%, not a claim of statistical significance.

Whole-record whitening fits AR coefficients on held-out-period signals, although it never uses their labels. It is an offline, transductive preprocessing recipe. In addition, `filtfilt` and `sosfiltfilt` use forward/backward filtering. These observations do not prove that the reported accuracy is inflated, but they prevent interpreting that result as prospective live decoding. Sources: [SciPy filtering documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html), [team method notes](team_submission_whole_record/METHOD_NOTES.md).

## New experiments actually run

Run `python3 explore_ecog_angles.py` from this folder. The script takes about 40 seconds in the checked local environment. Results are in `results/contribution_angles/`.

The new causal pipeline uses the same 60 ECoG channels, 1,200 Hz sampling, CAR, 1 Hz high-pass, 50 Hz harmonic notches, channel-wise AR(10), 50-300 Hz power, 250 ms bins, log10 transform, and shrinkage LDA. High-pass, notches, bandpass, and whitening now operate forward only. AR fitting uses 2.00-10.08 seconds of pre-task data, entirely before the first cue at 12.08 seconds. A future-truncation check confirms that removing subsequent samples does not change the earlier filtered signal.

This changes multiple signal-processing details together. Any observed gain cannot be attributed solely to causality or solely to calibration fitting. Neither cue values nor glove values enter the feature matrix. Cue onset still triggers the windows, so this is synchronous, cue-triggered replay, not detection of spontaneous movements.

| Pipeline / setting | Repeated random trial CV | Five purged temporal blocks |
| --- | ---: | ---: |
| Original offline classifier | 93.3% | 88.9%, 80/90 |
| Existing whole-record AR submission | 96.9% | 96.7%, 87/90 |
| Earlier offline pre-task AR | 96.9% | 97.8%, 88/90 |
| New causal pre-task AR, 60 channels, ends at 2.25 s | **97.4%** | **97.8%, 88/90** |
| New causal pre-task AR, 60 channels, ends at 1.00 s | **97.8%** | **97.8%, 88/90** |
| New causal pre-task AR, 20 selected channels, ends at 2.25 s | **96.7%** | **97.8%, 88/90** |

The new 1.00-second setting happens to have the same rounded 97.8% as the old submission's single-run result. They are different results: the former is a newly computed repeated-CV mean for a different pipeline; the latter is a single-run score. Label both settings and protocols whenever presenting them.

Random CV uses the exact previous 10-fold x 10-repeat splits (seed 0). Saved original and whitening fold scores were reproduced exactly before the new comparison. Temporal evaluation holds out five consecutive blocks of 18 trials and removes the immediately adjacent trial on each side from training. It tests within-recording time dependence, not a different session or participant. Random and blocked CV can still train on trials later than the test trial.

A separate expanding-window evaluation trains exclusively on the past:

| Earlier training trials | Next held-out trials | Correct |
| --- | --- | ---: |
| 1-44 | 46-60; trial 45 is a gap | 15/15 |
| 1-59 | 61-75; trial 60 is a gap | 14/15 |
| 1-74 | 76-90; trial 75 is a gap | 15/15 |

Total **44/45 correct, 97.8%**, using the full 2.25-second causal pipeline. Later fits can include previously tested trials once they become historical training data. This is an expanding-window replay protocol, not one permanently untouched 45-trial test set. It does not test the 1.00-second variant.

The causal high-gamma shuffled-label sanity check averaged 31.7% across 20 shuffles under blocked CV. This is near the 33.3% nominal chance level. A chance-level shuffle does not prove that all leakage, task confounds, or stimulus information have been eliminated.

### Candidate improvements that did not help

Six choices were fixed in code before this run. Candidate comparison uses the full 2.25-second causal features.

| Candidate | Repeated CV | Blocked CV |
| --- | ---: | ---: |
| High-gamma + shrinkage LDA | **97.4%** | **97.8%** |
| High-gamma relative to each trial's pre-cue power | 96.7% | 95.6% |
| High-gamma plus 13-30 Hz beta power | 97.3% | 94.4% |
| Ten channels selected in training folds | 95.1% | 94.4% |
| Twenty channels selected in training folds | 96.7% | 97.8% |
| Standardized high-gamma + linear SVM, C=1 | 95.7% | 93.3% |

Beta power is extracted before whitening. The relative-power candidate subtracts log pre-cue mean power (-1 to 0 s) from every post-cue bin. Channel ranking uses training-only ANOVA scores averaged over bins; selected channel identities vary by fold. SVM scaling is fitted on training data only.

Nested selection of these six candidates used three inner purged blocks inside each of five outer purged blocks. It reached **88/90, 97.8%** on outer predictions, selecting the 20-channel LDA in two folds and the full-channel LDA in three. It did not outperform the fixed causal LDA. Nested CV protects this specified selection process; it does not undo prior human exploration of this recording. Selecting whichever candidate looks best on outer scores would defeat the protection. Source: [scikit-learn nested CV documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### Interpret the engineering gains carefully

The short-window experiment uses samples from 0.25 to 1.00 s after the cue: 60 channels x 3 bins = 180 features, rather than 480. Its decision can have all required input samples **1.25 seconds earlier**, a 56% reduction in the cue-to-window-end waiting period. This is sample availability, not measured deployed response time, actual movement onset, or evidence that movement was predicted before it happened. Seven window lengths were examined; the best one remains an exploratory choice that needs independent validation.

Twenty channels x eight bins = 160 classifier inputs, two-thirds fewer than 480. CAR continues to use all 60 electrodes. This demonstrates reduced classifier input dimension; it does not demonstrate that only 20 implanted or acquired electrodes suffice. That would require rerunning the entire referencing and filtering pipeline with only the retained channels. The 1-second and 20-channel changes were tested separately; their combination was not tested.

The repeated full-window causal estimate is only +0.56 percentage points above the current submission's 96.9%. That small observed change is not established as statistically significant. The more useful contribution is the evidence that causal processing and shorter windows can retain strong accuracy in this recording.

## Contribution angles, in priority order

| Angle | Question and deliverable | What can be claimed now / what remains |
| --- | --- | --- |
| **1. Responsiveness** | Accuracy versus decision deadline; replay with cue and genuinely held-out prediction shown together | Curve generated. 1.00 s is promising. Independent validation and deployment timing remain. |
| **2. Temporal reliability** | Random CV, purged blocks, and past-only training compared side by side | Generated. All baseline fold scores reproduced; the causal replay succeeds on 44/45 forward trials. No new-subject claim. |
| **3. Signal economy** | Accuracy versus selected channels; show which channels recur across folds | Curve and selection lists saved. Reduced features tested. Reduced acquisition and channel-loss robustness remain. |
| **4. Failure explanation** | Inspect every recurrent cue/prediction mismatch using glove, electrode-time features, and held-out score contributions | New full-window causal repeated errors occur at trials 1 (90% of repeats), 16 (40%), and 65 (100%). Existing explanation figures can guide inspection, but refer to an older preprocessing condition. Do not relabel or remove difficult trials to raise accuracy. |
| **5. Know when to wait** | Calibrated confidence with an abstain option; plot error versus coverage and waiting time | Proposed. Calibrate inside training folds and select thresholds there. Report rejected trials and coverage; LDA scores or softmax alone are not demonstrated confidence calibration. |
| **6. Robustness and calibration cost** | Drop/noise-corrupt raw channels before CAR; measure learning curves with 5/10/20 training trials per class | Proposed. Use fixed tests and training-only selection. This can reveal practical limits even if headline accuracy decreases. |

An optional scientific extension is to distinguish decoding the requested cue from decoding executed posture. The glove is an execution audit, not an input to the ECoG classifier. Compare cue timing, glove movement onset, and model evidence on all trials. Visual cue and gesture identity are coupled in this dataset, so the present experiment alone cannot fully isolate motor information from stimulus-related information. The cheat sheet's "same movement needs less effort" interpretation should be phrased more cautiously as "lower high-gamma response despite similar glove posture" unless effort or energetics was measured.

## How to pursue more accuracy next

1. Preserve the established whitening submission; keep the new causal pipeline as a separate exploratory result.
2. Choose a primary goal before further search: 1-second prediction, reliability across time, or minimal features. Freeze the scoring protocol and a small candidate set.
3. If another classifier is justified, evaluate a faithful TVLDA implementation as a paper-informed next step. Simply flattening channel x time features into ordinary LDA is not TVLDA. The paper sums time-specific LDA evidence and includes a specific feature-reduction scheme. Source: [Gruenwald et al., 2019](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00901/full).
4. Another targeted feature test is splitting high-gamma into a few prespecified sub-bands, with all scaling and reduction learned within folds. Avoid an unrestricted band/window sweep on 90 trials.
5. Prefer a genuinely unseen recording if obtainable. All 90 trials here have now been repeatedly examined; reserving some of them afterward does not make a fresh blinded test.

Deep models are a lower priority with only 90 trials unless additional independent data becomes available. No accuracy target is promised.

## Presentation story and 140-second outline

Suggested title: **"Same movement, different past: fast ECoG decoding with honest validation."**

Map this outline into the official presentation template rather than inventing a competing submission format.

| Time | Message | Evidence |
| --- | --- | --- |
| 0-15 s | Team, ECoG category, requested gestures, project question | Team number visible throughout; actual names and location |
| 15-40 s | Established whitening improvement, 93.3% to 96.9% repeated CV | Matched preprocessing ablation; accurate labels on confusion matrix |
| 40-65 s | Team's existing brain-history findings | Abri's strongest verified carryover figure; attribute analysis |
| 65-105 s | Your addition: responsiveness and signal economy | New speed/channel curves, then a short held-out replay |
| 105-130 s | Show why the estimates are credible and bounded | Purged blocks; 44/45 past-only predictions; one honest error example |
| 130-140 s | What was learned and the next experiment | New recording, raw channel-loss test, calibrated abstention |

Leave 10 seconds below the official 2:30 maximum for editing and transitions. The organizer requests its template, the team number on screen throughout, the naming format `GroupNumber_GroupTopic_Groupname_Hostingplace`, and upload to the group's folder. The posted deadline is **5 October 2026, 11:30 am UTC-7**, which is **6 October 2026, 00:30 in Asia/Dhaka**. These are retrieved submission requirements, not authorization to upload. Source: [official README](https://github.com/unicorn-bi/Hackathon).

### Spoken draft

"We are [team name and number], working on the ECoG hand-pose challenge. We asked two connected questions: can we decode a gesture, and does the brain's recent history matter?

Our recording contains sixty brain-surface electrodes and ninety cued trials of rock, scissors, and paper. The glove helps us inspect execution, but the decoder uses only ECoG.

We first reproduced the original high-gamma classifier. Adding spectral whitening increased repeated cross-validation accuracy from 93.3 to 96.9 percent. The 97.8 percent confusion matrix is one run, so we report the repeated mean.

Our team's separate history analysis suggests that gesture identity fades after movement, while a broader movement-related state persists. [Show Abri's verified result and describe its measurement.]

We then asked what makes this decoder useful. In a separate replay experiment, whitening was calibrated before the task and every signal filter used only past samples. The full-window model achieved 97.4 percent repeated-CV accuracy. An exploratory window ending one second after the cue achieved 97.8 percent, with its required samples available 1.25 seconds earlier. Using twenty feature channels also retained strong performance.

We challenged these results with contiguous test blocks and models trained only on earlier trials. The full-window causal model classified forty-four of forty-five forward-test trials correctly. We retained difficult trials and saved every prediction.

These are results from one recording, not a live clinical system. Our contribution connects brain-history exploration with measurable responsiveness, transparent failures, and reproducible evaluation."

Adapt the history sentence to findings the team can substantiate; supporting scripts for several cheat-sheet claims were not supplied here. Rehearse the final recorded version to measure its actual duration.

## Reviewable outputs and checks

- `explore_ecog_angles.py`: runnable experiment code.
- `results/contribution_angles/metrics.json`: full metrics, exact preprocessing definitions, software versions, hashes, selected channels, nested choices, and limitations.
- `results/contribution_angles/experiment_arrays.npz`: causal features, coefficients, fold IDs, and predictions.
- `results/contribution_angles/trial_predictions.csv`: requested cue, held-out first-repeat prediction, correctness, and repeated error frequency for all 90 trials. Supports a replay without training on the displayed trial.
- `results/contribution_angles/speed_and_channels.png`: two measured engineering tradeoffs.
- `results/contribution_angles/validation_comparison.png`: temporal evaluation comparison.

Verification completed: original/whitening fold scores reproduced; every trial tested once per repeated-CV repetition; no train/test overlap; temporal neighbors purged; channel ranking and SVM scaling fitted in training only; causal future-truncation check passed; protected source hashes unchanged; Python compilation passed; generated figures visually inspected.

## Prepared team update

**English:** I explored a complementary angle: how quickly and with how many feature channels we can decode the gestures. In a separate causal replay pipeline with whitening fitted before the first cue, full-window repeated-CV accuracy was 97.4%. A window ending 1.0 s after the cue reached 97.8%, while using 20 feature channels gave 96.7%. These are exploratory results on the same recording, so I would keep 96.9% as the established submission number and present the new curves as engineering findings. I also tested chronological blocks and past-only training: the full-window causal pipeline got 44/45 forward-test trials correct. The original submission is unchanged; code, predictions, and plots are saved locally.

**한국어:** 기존 연구를 보완하는 방향으로, 얼마나 빨리 그리고 몇 개 전극의 특징으로 손동작을 구분할 수 있는지 확인했어요. 첫 지시 전에 whitening 필터를 추정하고 과거 샘플만 사용하는 별도의 처리 방식에서, 전체 시간 구간의 반복 교차검증 정확도는 97.4%였어요. 지시 후 1.0초까지만 사용하면 97.8%, 전극 특징 20개를 사용하면 96.7%였어요. 모두 같은 기록에 대한 탐색 결과이므로, 기존 제출 결과는 96.9%로 유지하고 새 그래프는 응답 시간과 특징 수에 대한 추가 실험으로 발표하는 것을 제안해요. 시간순 구간 평가와 과거 동작 기록만으로 학습하는 평가도 했고, 전체 구간을 쓰는 방식은 미래 테스트 동작 45개 중 44개를 맞혔어요. 원래 제출 코드는 그대로이며, 코드와 예측값, 그림은 로컬에 저장했어요.

This update is prepared for review; it has not been sent.
