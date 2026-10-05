# Team update draft — not sent

The project now centers on whitening as our strongest supported preprocessing direction. We reproduced ECoG_G25_code's causal-filter result: 94.4% → 97.3%, 11 trials with improved 20-seed correctness fraction, 0 worsened, one-sided sign-flip p=0.00044998 (rounds to 0.0004).

We distinguish the whole-record AR fit behind that headline from a filter fitted only before the held-out period. The original causal first-45 → last-45 check improves 39/45 → 40/45. Pre-task-only and purged time-midpoint checks are now exported beside the source reproduction. First difference also reaches 97.3%, so we avoid claiming AR fitting is uniquely responsible.

The still-hand-versus-Paper interpretation is withdrawn as a central claim after the reported failed follow-up. Movement history, glove behavior and unusual cases remain secondary diagnostics. The established 96.9% submission stays intact. Dashboard, figures, report, 140-second script and portable team bundles have been refocused. Work stays local; this draft has not been sent.

한국어: 핵심 기여를 whitening의 검증으로 바꿨습니다. causal filter 결과 94.4% → 97.3%와 동일 CV 조건의 개선 trial 11개, 악화 0개를 재현했습니다. 전체 기록으로 AR을 학습한 결과와 시험 구간 이전에 학습을 끝낸 결과를 구분합니다. first-difference도 비슷한 성능이어서 AR만의 고유한 효과라고 주장하지 않습니다. still-hand/Paper 가설은 핵심 주장으로 사용하지 않으며, 기존 96.9% 제출본과 원본 데이터는 보존했습니다. 아직 팀에 전송하지 않았습니다.
