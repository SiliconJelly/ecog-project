# Abri에게 보낼 결과 공유 초안

**이 파일은 첫 휴식 구간 방식의 이전 초안이다. 새 팀 지침의 전체 기록 방식과는 다르다. 최신 제출 코드와 결과 공유 초안은 `team_submission_whole_record/README.md`를 사용한다. 기존 내용을 새 지침 결과로 전송하지 않는다.**

아래는 전송하지 않은 초안이다. 코드와 결과를 함께 검토한 뒤 필요한 부분을 공유하면 된다.

## English

I reproduced your original `classify.py` result without changing the file: 93.3% ± 7.4% for the three gestures.

I then added per-channel AR(10) spectral whitening after the notch filters and before the 50–300 Hz bandpass. With the same 90 trials, 480 features, shrinkage LDA, and identical 10-fold × 10 repeated CV splits, accuracy increased to **96.9% ± 5.7%**, an improvement of **3.56 percentage points**. Fist versus peace increased from 93.5% to 96.7%. The shuffled-label control was **33.3%**, close to chance.

For the whitening coefficients, I used only the initial cue-free period: I preprocessed 0–12.08 s separately and fitted the AR models on 2.00–10.08 s. That keeps all 90 task trials out of the filter-fitting data. The order, calibration interval, features, and CV seed were fixed before evaluating whitening accuracy.

In the first 10-fold run, the confusion matrix improved from 84/90 to 87/90 correct, with 29/30 correct for each gesture. These single-run counts are separate from the repeated-CV mean. The standard deviations above are across fold scores, not confidence intervals.

I saved the comparison code, whitening coefficients, split indices, predictions, and before/after spectrum and confusion-matrix plots. This gives us a controlled whitening comparison within your baseline. It is an improvement on this recording; we have not tested a new participant or session.

## 한국어

파일을 수정하지 않고 원래 `classify.py`를 실행해서, 세 동작 분류 정확도 93.3% ± 7.4%를 재현했어요.

그다음 notch 필터 뒤, 50–300 Hz bandpass 앞에 채널별 AR(10) spectral whitening을 추가했어요. 동일한 90개 동작 기록, 특징 480개, shrinkage LDA, 그리고 완전히 같은 10-fold × 10회 반복 분할을 사용했을 때 정확도가 **96.9% ± 5.7%**로 높아졌어요. **3.56%p 개선**된 결과예요. 바위와 가위만 구분한 정확도는 93.5%에서 96.7%로 높아졌고, 정답을 섞은 대조 검사에서는 우연 수준에 가까운 **33.3%**가 나왔어요.

Whitening 계수에는 최초 동작 지시 전의 구간만 사용했어요. 0–12.08초를 따로 전처리하고 2.00–10.08초에서 AR 모델을 맞췄기 때문에, 동작 기록 90개는 필터 계수 추정에 들어가지 않았어요. AR 차수, 보정 구간, 특징, CV 시드는 whitening 정확도를 보기 전에 정했어요.

첫 번째 10-fold 검사에서는 정답 수가 84/90에서 87/90으로 늘었고, 각 동작은 30번 중 29번을 맞혔어요. 이 한 차례의 정답 수는 반복 CV의 평균과 구별해야 해요. 위 표준편차는 시험 묶음별 점수의 차이이며 신뢰구간은 아니에요.

비교 코드, whitening 계수, 학습·시험 인덱스, 예측값, 처리 전후 스펙트럼과 혼동행렬 그림을 저장했어요. 기존 분석 안에서 whitening의 효과를 비교한 결과이고, 이 기록에서는 개선이 관찰됐어요. 새로운 참가자나 다른 세션에서의 성능은 아직 검사하지 않았어요.

## 공유할 파일

- `whitening_compare.py`
- `results/whitening/comparison_metrics.json`
- `results/whitening/comparison_arrays.npz`
- `results/whitening/whitening_spectrum.png`
- `results/whitening/accuracy_comparison.png`
- `results/whitening/confusion_comparison.png`

재실행에는 기존 데이터 `source/ECoG_Handpose.mat`와 원본 실행 결과 `results/baseline_results.npz`도 필요하다. 원본 실행 결과의 수치 기록은 `results/baseline_metrics.json`에 있다.
