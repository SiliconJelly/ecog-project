# Core classifier submission with whole recording whitening

## English

This is the completed **core `classify.py` whitening comparison** requested in the new team instructions. It is not the initial-rest-only whitening experiment. The bonus rest/carryover scripts are not included in this submission.

The code uses one AR(10) Yule–Walker prediction-error FIR per electrode, fitted on all 507,025 continuous post-notch samples. It applies `[1, -a1, ..., -a10]` once using `scipy.signal.lfilter`, after the original notch loop and before the original 50–300 Hz bandpass. The 480 features, shrinkage LDA, CV splits and seeds are unchanged.

### Results

| Measure | WHITEN = False | WHITEN = True |
| --- | --- | --- |
| Three-class accuracy, mean ± fold SD | 93.3% ± 7.4% | 96.9% ± 5.5% |
| Fist vs peace, mean ± fold SD | 93.5% ± 9.4% | 97.2% ± 6.7% |
| Shuffled-label control | 33.8% | 33.7% |
| Correct in the first complete 10-fold evaluation | 84/90 | 88/90 |

The repeated-CV three-class mean improved by **3.56 percentage points**. The exact means are 93.333333% and 96.888889%. The first evaluation's 88/90 is a separate result, not the repeated-CV mean.

Confusion matrices, true rows and predicted columns, ordered **Fist, Peace, Open**:

```text
WHITEN = False       WHITEN = True
28  2  0             29  1  0
 2 28  0              0 30  0
 0  2 28              0  1 29
```

The off result reproduced the original features and three-class fold scores exactly. Original code and data were not modified. The real printed output is supplied verbatim, not reconstructed from this table.

### Files to review

- `classify_whitened.py`: standalone submitted script; `WHITEN = True` at line 10.
- `results/off/printed_output.txt`: actual output with whitening off.
- `results/on/printed_output.txt`: actual output with whitening on.
- `results/on/channel26_whitening_spectrum.png`: requested channel 26 before/after Welch spectra on logarithmic axes.
- `results/off/confusion_matrix.png` and `results/on/confusion_matrix.png`: original-style confusion plots.
- `results/off/metrics.json` and `results/on/metrics.json`: precise results, versions and source hashes.
- `diff_vs_original.patch`: all differences from the original classifier.
- `METHOD_NOTES.md`: explanation of every added part, with Korean translation.
- `validation.json`: verification checks and spectrum measurements.

SVG versions of the plots are also included. The raw MAT recording and derived trial-feature archives are deliberately **not** bundled. They are not required to read the output; the original MAT file is required to rerun the script.

### Run the submitted script

Place the already-shared `ECoG_Handpose.mat` in this folder and run from this folder:

```bash
python classify_whitened.py
```

Dependencies are `numpy`, `scipy`, `matplotlib`, and `scikit-learn`. To verify your own environment, set line 10 to `WHITEN = False`, run and compare with the supplied off output. Then set it back to `True` and run. No local absolute path or existing baseline-result archive is needed by the submitted script.

On hhlee's computer, the verified local recorder can rerun both conditions without editing the submitted file or copying the MAT file:

```bash
/Users/hhlee/miniconda3/envs/ecog/bin/python /Users/hhlee/hackathon_project/run_team_submission.py --whiten off
/Users/hhlee/miniconda3/envs/ecog/bin/python /Users/hhlee/hackathon_project/run_team_submission.py --whiten on
```

### Limitations

This uses one recording and the team's whole-recording **offline** filter fit, including evaluation-period signal values without labels. It does not establish future-data, real-time, independent-session or participant generalization. Repeated fold scores are not independent experiments and their SD is not a confidence interval.

The channel 26 spectrum is flatter after whitening, but not perfectly flat. Its 50–300 Hz spectral flatness is approximately 0.203 → 0.272, excluding ±3 Hz around the notch frequencies. No additional parameter tuning was used to make the spectrum or accuracy look better.

The earlier initial-rest-only method also had a three-class mean of 96.9%, but its SD was 5.7%, its binary mean was 96.7%, and its first confusion matrix had 87/90 correct. That is a distinct experiment. The two means happen to coincide; their features, filters and fold scores are different.

### Unsent message draft

I matched the whole-recording whitening recipe in your README. With whitening off, the original result is reproduced exactly: 93.3% ± 7.4%. With AR(10) whitening on, three-class accuracy is 96.9% ± 5.5%, fist vs peace is 97.2% ± 6.7%, and the shuffled control is 33.7%. The first confusion matrix has 88/90 correct. I prepared the script, verbatim output, and channel 26 spectrum plot. This is the core classifier result; I haven't completed the bonus scripts yet.

## 한국어

이번 파일은 새 팀 지침에 맞춘 **핵심 분류 코드의 완성본**입니다. 기존 최초 휴식 구간 방식과는 별도입니다. 보너스인 휴식·이전 동작 영향 분석 스크립트는 아직 포함하지 않았습니다.

원본 notch 처리 뒤의 전체 507,025개 샘플로 전극마다 10차 AR 계수를 Yule–Walker 방식으로 추정했습니다. `[1, -a1, ..., -a10]`을 `lfilter`로 한 방향 적용한 다음 기존 bandpass를 사용합니다. 특징 480개, LDA, 평가 방식과 시드는 바꾸지 않았습니다.

위 결과표에서 왼쪽은 whitening을 끈 결과, 오른쪽은 켠 결과입니다. 세 동작 반복 평가 평균은 **93.3% ± 7.4% → 96.9% ± 5.5%**, 개선은 **3.56%p**입니다. 바위·가위는 **93.5% → 97.2%**, 정답 섞기 결과는 **33.7%**로 우연 수준에 가깝습니다. 첫 전체 평가의 정답 수는 **84/90 → 88/90**이며, 반복 평가 평균과 구별합니다.

공유할 것은 이 폴더의 `classify_whitened.py`, 실제 출력 두 개, ECoG 26번 전극 스펙트럼, 혼동행렬과 설명입니다. 같은 내용을 `../ecog_team_submission_whole_record.zip`에 묶습니다. 데이터 원본과 trial 특징 배열은 압축파일에 넣지 않습니다. 제출 코드의 재실행에는 팀이 이미 공유한 MAT 파일만 있으면 됩니다.

팀원은 압축을 풀고 이 폴더에 MAT 파일을 둔 다음, 이 폴더에서 `python classify_whitened.py`를 실행하면 됩니다. `WHITEN` 스위치는 10번째 줄입니다. 먼저 `False`로 실행해 원본 출력과 비교하고, 그다음 `True`로 실행합니다. 추가한 부분의 자세한 설명은 `METHOD_NOTES.md`에 있습니다.

전체 기록으로 필터를 만든 오프라인 평가라는 한계는 그대로 남습니다. 필터 추정에는 정답 라벨을 쓰지 않지만 평가 구간의 신호는 사용합니다. 스펙트럼도 완전히 평평해진 것은 아닙니다. 파라미터를 바꿔 더 좋아 보이게 만들지는 않았습니다.

이전 결과와 새 결과의 세 동작 평균은 우연히 같습니다. 그러나 기존 첫 휴식 구간 방식은 표준편차 5.7%, 바위·가위 평균 96.7%, 첫 평가 정답 87/90이었고, 새 전체 기록 방식은 각각 5.5%, 97.2%, 88/90입니다. 이전 결과나 그림은 덮어쓰지 않았습니다.

### 위 영어 메시지의 한국어 번역

README에 적힌 전체 기록 whitening 방식으로 맞췄어요. Whitening을 끄면 원래 결과인 93.3% ± 7.4%가 정확히 재현돼요. AR(10) whitening을 켜면 세 동작 정확도는 96.9% ± 5.5%, 바위·가위는 97.2% ± 6.7%, 정답 섞기 결과는 33.7%예요. 첫 혼동행렬에서는 90회 중 88회를 맞혔어요. 코드, 실제 출력과 26번 전극 스펙트럼을 준비했어요. 이번 것은 핵심 분류 결과이고, 보너스 스크립트는 아직 완료하지 않았어요.

이 메시지는 초안이며 전송하거나 업로드하지 않았습니다.
