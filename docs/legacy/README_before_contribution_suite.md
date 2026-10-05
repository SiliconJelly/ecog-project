# BR41N.IO / IEEE SMC 2026 해커톤 작업 메모

작성일: 2026-10-05

이 파일은 대화를 이어갈 때 참고하는 작업 메모다. 전체 대화 기록은 아니며, 진행에 따라 역할·결과·다음 작업을 갱신한다.

## 현재 완료된 결과

**새 팀 지침에 맞춘 핵심 분류 제출본:** [team_submission_whole_record/README.md](team_submission_whole_record/README.md). 전체 연속 기록으로 AR(10) 필터를 추정했고, 끈 상태의 원본 재현을 검증한 뒤 켠 상태를 실행했다. 결과는 **93.3% ± 7.4% → 96.9% ± 5.5%**, 바위·가위는 **97.2% ± 6.7%**, 정답 섞기는 **33.7%**다. 코드·실제 출력·스펙트럼·혼동행렬을 `ecog_team_submission_whole_record.zip`에 준비한다. 아직 업로드하지 않았고 보너스 분석은 포함하지 않는다.

아래의 기존 비교와 팀원 시각화는 **첫 휴식 구간으로 필터를 추정한 이전 실험**이다. 새 전체 기록 방식과 평균이 같아도 조건과 결과 배열은 다르며, 기존 결과를 덮어쓰지 않는다.

Whitening 전후 비교를 완료했다. 동일한 90회 동작과 10-fold × 10회 반복 분할에서 세 동작 평균 정확도는 **93.3% ± 7.4% → 96.9% ± 5.7%**, 차이는 **+3.56%p**다. Whitening의 정답 섞기 대조 결과는 33.3%다.

완료된 결과는 [결과 보기 노트북](02_results_view.ipynb)의 첫 표부터 보면 된다. 기존 분석 노트북의 열린 탭에는 준비 단계의 내용이 남아 있을 수 있어 결과 전용 노트북을 별도로 만들었다. [상세 결과](WHITENING_RESULTS.md)에는 구현·평가·그림 해석을 적었고, [팀원 공유 초안](TEAM_UPDATE.md)에는 영어와 한국어를 함께 준비했다. 메시지는 전송하지 않았다.

## 목표와 역할

- 팀: g25, 현재 공유된 대화에서는 hhlee와 Abri가 먼저 협업을 시작했다.
- 과제: ECoG 신호로 가위·바위·보 손동작을 분석한다.
- hhlee 담당: Abri의 기존 분류 코드에 spectral whitening을 추가하고, 추가 전후 성능을 같은 조건으로 비교한다.
- Abri 담당: 휴식 구간의 이전 동작 흔적, temporal generalization, 전극 간 coupling 분석.
- G55 발표자료와 Gruenwald et al. (2019) 논문은 참고 자료로 사용한다.

## 함께 작업하는 방식

- 대화는 현재 채팅에서 이어간다. 이 파일을 만든다고 새 채팅이 자동으로 생기지는 않는다.
- 해커톤 코드와 데이터 작업 위치는 `/Users/hhlee/hackathon_project`다.
- `/Users/hhlee/ecog_project`는 기존 캡스톤·연습 자료를 두는 별도 폴더로 유지한다.
- 코딩과 실행은 어시스턴트가 직접 수행한다. 사용자에게 반복적인 복사·붙여넣기를 요구하지 않는다.
- 작업 전에 무엇을 왜 하는지 설명하고, 작업 후에는 결과와 의미를 쉬운 말로 설명한다.
- 한 단계씩 진행하며, 맡은 실험에서 임의로 여러 모델이나 새로운 연구 방향으로 확장하지 않는다.
- 원본 자료는 보존하고, 분석에 필요한 작업용 사본을 사용한다.
- 영어 메시지를 제안할 때는 한국어 번역도 함께 제공한다.

## 파일과 환경

- Python: `/Users/hhlee/miniconda3/envs/ecog/bin/python`
- 분석 노트북: `/Users/hhlee/hackathon_project/01_baseline_whitening.ipynb`
- 결과 보기 노트북: `/Users/hhlee/hackathon_project/02_results_view.ipynb`
- 직접 재실행하는 노트북: `/Users/hhlee/hackathon_project/03_run_whitening.ipynb`. 코드 셀 하나를 실행하면 새로 계산한 결과표와 그림 3개가 표시된다.
- 가위·바위 구분 설명 노트북: `/Users/hhlee/hackathon_project/04_decoding_explained.ipynb`. 실행 없이 전극·시간 패턴과 실제 테스트 판정 점수를 읽는다.
- 해커톤 작업용 자료: `/Users/hhlee/hackathon_project/source`
- Whitening 구현: `/Users/hhlee/hackathon_project/whitening_compare.py`
- 비교 결과: `/Users/hhlee/hackathon_project/results/whitening`
- 팀원 공유용 시각화: `/Users/hhlee/hackathon_project/results/team_figures`. `README.md`에 그림 설명과 영어·한국어 캡션이 있다.
- 공유 압축파일: `/Users/hhlee/Downloads/Hackathon-20261004T203725Z-1-001.zip`
- 이미 압축이 풀린 공유 폴더: `/Users/hhlee/Downloads/Hackathon-20261004T203725Z-1-001/Hackathon`
- 참고 발표자료: `/Users/hhlee/Downloads/br41n.io_g55.pptx`
- 이전 연습 데이터: `/Users/hhlee/ecog_project/data/fingerflex`

공유 폴더에는 다음 9개 파일이 있다.

| 파일 | 내용 |
| --- | --- |
| `ECoG_Handpose.mat` | 손동작 데이터 |
| `data_description.pdf` | 실험 설명, 채널 구성, 전극 배치 |
| `look.py` | ECoG·동작 지시·손가락 센서 시각화 |
| `high_gamma.py` | 동작별 high-gamma 활동 비교 |
| `classify.py` | 비교 출발점이 되는 high-gamma + LDA 분류 코드 |
| `baseline_state.py` | 휴식 구간과 이전 동작의 영향 분석 |
| `Hackathon.docx` | 팀원의 결과 기록과 그림 |
| `Copy of Hand.pdf` | Gruenwald et al. (2019) TVLDA·spectral whitening 논문 |
| `join_mat.py` | 분할 파일 재결합 도구. 현재 폴더에서는 실행하지 않는다. |

`join_mat.py`는 출력 파일을 쓰기 모드로 먼저 연다. 현재 공유 폴더에는 `.part` 파일이 없으므로, 실행하면 완성된 `ECoG_Handpose.mat`를 빈 파일로 덮어쓴다.

## 직접 확인한 데이터

- MATLAB 변수: `y`, 크기 `(67, 507025)`.
- 행 1: 시간. 약 422.52초, 샘플링 주파수 1200 Hz.
- 행 2–61: ECoG 전극 신호 60개.
- 행 62: 동작 지시. `0` 휴식, `1` 바위, `2` 가위, `3` 보.
- 행 63–67: 엄지·검지·중지·약지·새끼손가락 센서.
- 각 동작 지시 30회, 총 90회.
- 동작 지시는 실제 움직임 시작 시점과 구별한다. 손가락 센서로 실제 움직임을 함께 확인할 수 있다.
- 이전의 `fingerflex`와 별도의 데이터다.

## 팀원 코드와 기록된 결과

`classify.py`의 흐름은 CAR → 1 Hz high-pass → 50 Hz 및 고조파 notch → 50–300 Hz bandpass → log power 특징 → shrinkage LDA다.

- 동작 지시 시작 후 0.25–2.25초를 0.25초 구간 8개로 나눈다.
- 60채널 × 8구간 = 480개 특징, 총 90개 trial을 사용한다.
- 평가: 10-fold cross-validation을 10회 반복한다.
- 팀원 문서의 기록: 세 동작 정확도 `93.3% ± 7.4%`, shuffled-label 결과 `33.8%`.
- 어시스턴트가 `ecog` 환경에서 작업용 `classify.py`를 수정하지 않고 한 번 실행했다. 세 동작 정확도 `93.3% ± 7.4%`, 바위·가위 정확도 `93.5% ± 9.4%`, shuffled-label 결과 `33.8%`로 팀원 기록과 일치했다.
- 한 번의 10-fold 평가로 만든 혼동행렬은 `[[28, 2, 0], [2, 28, 0], [0, 2, 28]]`다. 각 동작 30회 중 28회를 맞혔다. 이 그림과 반복 평가 평균은 구별한다.
- 실행 결과는 `results/baseline_results.npz`, `results/baseline_metrics.json`, `results/baseline_confusion_matrix.png`에 저장했다. 실행 전후 코드의 SHA-256이 같아 파일을 수정하지 않은 것을 확인했다.
- 공유 해커톤 자료 9개의 사본을 `hackathon_project/source`에 준비했다. 각 사본이 원본과 동일한지 SHA-256으로 확인했다.
- 처음 준비한 `ecog_project/hackathon` 폴더를 `/Users/hhlee/hackathon_project`로 옮겨 캡스톤 작업과 분리했다. 작업 메모도 이 폴더의 `README.md`로 옮겼다.
- 분석 노트북에 실제 실행 결과를 설명하는 Markdown과 저장한 그림을 추가했다. 코드는 터미널에서 실행했으며, 준비된 노트북 코드 셀의 실행 횟수와 출력은 비워 두었다.

## Whitening 구현과 검증

- 채널별 AR(10) Yule-Walker 예측 오차 FIR 필터를 notch 뒤, high-gamma bandpass 앞에 추가했다.
- 최초 동작 지시 전인 0–12.08초 신호를 별도로 전처리하고, 양끝 2초를 뺀 2.00–10.08초에서 필터 계수를 추정했다. 동작 90회의 자료와 정답은 계수 추정에 사용하지 않았다.
- 특징 480개, shrinkage LDA, CV 분할을 유지했다. 기존 특징의 최대 차이는 0.0이고, 기존 100개 시험 점수 및 첫 반복 예측이 원본 실행과 정확히 일치했다.
- Whitening 후 세 동작 정확도는 96.9% ± 5.7%, 바위·가위 정확도는 96.7% ± 7.1%, 정답을 섞은 결과는 33.3%다.
- 첫 반복에서는 정답 수가 84/90에서 87/90으로 늘었다. 기존 오답 4개를 고치고 새 오답 1개가 생겼다.
- AR 차수와 보정 구간은 점수를 보기 전에 고정했다. 그림의 글자 겹침을 고친 뒤 같은 설정으로 재실행했으며, 결과 수치를 바꾸기 위한 설정 탐색은 하지 않았다.
- 한 사람의 한 기록에 대한 오프라인 CV 결과다. 다른 참가자·세션, 실시간 처리, 통계적 유의성을 검증한 결과로 표현하지 않는다. 반복 CV의 표준편차는 신뢰구간이 아니다.
- 그림, 수치, 계수, 특징, 학습/시험 인덱스, 예측값을 `results/whitening/`에 저장했다.

G55는 겹치는 1.5초 구간을 사용한 5-fold 평가에서 Macro-F1 99.2%, 별도 시험 정확도 98.7%를 보고한다. 팀원의 trial 단위 평가와 조건·지표가 다르므로 수치를 그대로 비교하지 않는다. G55 구현 코드는 이번 공유 폴더에 없다.

## 작업 상태와 다음 대화

팀원의 기준 결과 재현과 whitening 전후 비교를 완료했다.

1. 완료: `hackathon_project/source`에 작업용 사본을 준비했다.
2. 완료: whitening을 추가하기 전에 기존 분류 코드를 실행하고 결과를 기록했다.
3. 완료: notch 뒤, high-gamma bandpass 앞에 채널별 10차 spectral whitening을 추가했다.
4. 완료: 특징 추출·LDA·평가 조건과 동일한 CV 분할로 추가 전후를 비교했다.
5. 완료: 평균·표준편차, 혼동행렬, shuffled-label 결과와 주파수별 신호 분포를 저장했다.
6. 완료: 사용자가 읽을 노트북과 결과 보고서, 팀원에게 보낼 영어·한국어 초안을 준비했다.

다음 대화에서는 정확도 비교 그림과 혼동행렬을 함께 읽고, 팀원에게 어떤 결과를 전달할지 정하면 된다. 사용자가 요청하기 전에는 새로운 모델이나 추가적인 설정 탐색으로 분석 범위를 넓히지 않는다.
