# Whole recording whitening implementation

## English

This submission follows the new team recipe in `ecog_team_whitening.zip/README_FOR_TEAM.md`. The only signal-processing addition to the original `classify.py` is per-channel AR(10) whitening after the existing notch filters and before the unchanged 50–300 Hz bandpass. The original feature extraction, LDA, CV splits, seeds, shuffled-label control, and confusion-matrix computation remain unchanged.

### What the added lines do

- `WHITEN = True` enables whitening. Change only this switch to `False` to reproduce the original decoder.
- `solve_toeplitz` solves the Yule–Walker equations using the autocovariance's Toeplitz structure.
- `order = 10` uses the previous ten samples in each channel's autoregressive model.
- `sample_count = channel_signals.shape[1]` uses all 507,025 continuous samples, including rest and task periods. There is no initial-rest-only calibration or trimming.
- `centered = channel_signal - channel_signal.mean()` removes the channel's mean for coefficient estimation, not an additional filter on the classifier input.
- Each covariance entry is a lagged dot product divided by the same total sample count. These are biased, demeaned autocovariance estimates for lags 0–10.
- `covariance /= covariance[0]` normalizes the equations for numerical scale. It does not change the fitted AR coefficients.
- `predictor = solve_toeplitz(...)` estimates the ten coefficients that predict a sample from its past values.
- `np.r_[1.0, -predictor]` constructs the prediction-error FIR coefficients `[1, -a1, ..., -a10]`.
- `ss.lfilter(..., [1.0], channel_signal)` applies that FIR once in the forward direction to the entire post-notch channel. Whitening does not use `filtfilt`.
- `ss.welch(ecog[25], ...)` estimates channel 26's power spectrum immediately before and after whitening, with the same 2-second Welch segments. This plot does not affect the classifier.
- `loglog(...)` displays both frequency and spectral power on logarithmic axes. The shaded region is the unchanged classifier band, 50–300 Hz.
- The extra whitening print identifies the recipe and sample count. All original result-printing lines remain unchanged.

The zero-phase high-pass, notch, and bandpass filters already present in the original code are retained. The one-pass requirement applies to the newly added whitening FIR, not to those existing filters.

### Verification and interpretation

With whitening off, the feature matrix, 100 three-class fold scores, first complete 10-fold predictions, shuffled scores, binary accuracy summary, and confusion matrix must match the recorded baseline. The off result is checked before the on result is allowed to run.

The local recorder changes the switch in memory, captures the script's printed output verbatim, and saves figures instead of opening GUI windows. It does not edit the submitted script or its learning/evaluation computations. The submitted script defaults to `WHITEN = True`.

Whitening fitting uses signal values, not gesture labels. Under the team's whole-recording recipe it includes the signals from evaluation periods. This is an offline within-recording comparison, not a demonstration of prospective real-time decoding or independent-session generalization. Repeated-CV SD is not a confidence interval.

The earlier 96.9% result used initial-rest-only calibration, so it remains a separate experiment. This submission does not overwrite that result. The bonus scripts are not included in this core classifier submission.

## 한국어

이번 제출본은 새 팀 지침의 **전체 연속 기록으로 필터를 추정하는 방식**을 따릅니다. 기존 `classify.py`의 notch 뒤, 기존 50–300 Hz bandpass 앞에 전극별 AR(10) whitening만 추가했습니다. 특징 추출, LDA, 교차검증 분할과 시드, 정답 섞기 검사, 혼동행렬 계산은 원본 그대로입니다.

### 추가한 각 부분의 의미

- `WHITEN = True`: whitening을 켭니다. `False`로 바꾸면 원래 분류 방식입니다.
- `solve_toeplitz`: 신호의 자기공분산으로 Yule–Walker 방정식을 풉니다.
- `order = 10`: 각 전극에서 직전 10개 샘플로 현재 샘플을 예측하는 모델입니다.
- `sample_count`: 쉬는 구간과 동작 구간을 포함한 전체 507,025개 샘플을 사용합니다. 처음 8초만 사용하거나 가장자리를 잘라내지 않습니다.
- `centered`: 계수를 추정할 때 해당 전극의 평균을 뺍니다. 분류 입력에 별도 필터를 추가하는 것이 아닙니다.
- `covariance`: 시간차 0–10의 자기공분산을 계산합니다. 각 시간차의 합을 같은 전체 샘플 수로 나누는 biased 추정입니다.
- 자기공분산을 첫 값으로 나누는 부분: 계산의 수치 규모를 맞추며, AR 계수 자체를 바꾸지 않습니다.
- `predictor`: 직전 10개 샘플의 예측 계수를 추정합니다.
- `[1.0, -predictor]`: 현재 신호에서 예측 가능한 부분을 빼는 FIR 필터 계수입니다.
- `lfilter`: 전체 notch 처리 신호에 이 필터를 앞 방향으로 한 번 적용합니다. whitening에 `filtfilt`를 쓰지 않습니다.
- `welch`: ECoG 26번 전극의 처리 전후 주파수별 파워를 같은 조건으로 계산합니다. 그림 계산은 분류 결과에 영향을 주지 않습니다.
- `loglog`: 주파수와 파워를 로그 축으로 표시합니다. 회색 부분은 기존 분류 대역 50–300 Hz입니다.
- whitening 상태를 출력하는 줄: 어떤 방식과 샘플 수를 사용했는지 남깁니다. 기존 결과 출력 줄은 그대로입니다.

원래 있던 high-pass, notch, bandpass의 양방향 필터는 유지했습니다. 새 지침의 한 방향 적용은 새로 넣은 whitening 필터에 대한 조건입니다.

### 검증과 해석 범위

끄고 실행한 결과부터 원본과 비교합니다. 특징값, 세 동작 시험 점수 100개, 첫 전체 평가의 예측, 정답 섞기 점수, 바위·가위 정확도 요약과 혼동행렬을 검증한 뒤 켜고 실행합니다.

로컬 기록 도구는 스위치만 메모리 안에서 바꾸고, 실제 출력 문장을 그대로 저장하며, 화면을 띄우는 대신 그림을 저장합니다. 제출 코드 파일이나 학습·평가 계산을 수정하지 않습니다. 제출본의 기본값은 `WHITEN = True`입니다.

필터 계수 추정에는 동작 정답을 쓰지 않습니다. 다만 전체 기록 방식에는 평가 구간의 신호도 들어갑니다. 따라서 한 기록 안에서 한 오프라인 비교이며, 미래 신호를 모르는 실시간 사용이나 새 세션에서 검증한 결과는 아닙니다. 반복 평가의 표준편차는 신뢰구간이 아닙니다.

예전 96.9%는 첫 휴식 구간으로 필터를 만든 별도 실험입니다. 이번 결과로 덮어쓰지 않습니다. 이번 핵심 분류 제출본에는 보너스 분석 스크립트가 포함되지 않습니다.
