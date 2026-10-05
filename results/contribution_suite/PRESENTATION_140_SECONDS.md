# 140-second presentation guide: Whitening Under Test

Updated 6 October 2026. Display Team G25 and the verified team identity; use the team's actual submission template. Rehearse the spoken timing.

| Time | Visual | Message |
|---|---|---|
| 0–15 s | Question and pipeline | Can the same recorded signal yield a more reliable gesture decision? |
| 15–40 s | Whitening evidence | Reproduced 94.4% → 97.3%; matching seeds and paired trial changes |
| 40–65 s | Calibration timeline | Causal filtering and earlier-signal calibration are separate |
| 65–90 s | Mechanism control | First difference ties AR; avoid claiming unique mechanism |
| 90–115 s | Speed / confidence / replay | Earlier decisions, visible mistakes and abstentions |
| 115–140 s | Limits and contribution | Withdraw unsupported still-hand/Paper claim; test new recordings next |

Spoken draft:

We are Team G25, decoding rock, scissors and paper from sixty ECoG channels across ninety trials. Our strongest supported processing direction is spectral whitening. We ask whether it improves decisions when the rest of the pipeline stays the same.

Whitening predicts a channel's next sample from its recent history, then subtracts that predictable component. This changes the frequency balance before we measure high-gamma power. It adds no new recorded information.

We reproduced the team's follow-up scripts. With forward-only filters, average gesture accuracy rises from 94.4 to 97.3 percent. On matching cross-validation seeds, 11 trials improve their fraction correct and none worsen. The original one-sided sign-flip test rounds to p equals 0.0004.

That headline uses a filter fitted on the whole recording. We therefore distinguish causal application from past-only calibration. Fitting through the first forty-five trials and testing the last forty-five gives 39 correct without whitening and 40 with it. This is a modest gain, not prospective ninety-seven-percent accuracy.

We also test pre-task-only calibration, purged chronological blocks, earlier decision deadlines and lost channels. At one second our whitened model gets 88 of ninety correct. A confidence policy accepts 77 trials, gets 75 right, and abstains on 13. Recorded replay exposes the mistakes.

A fixed first-difference filter ties AR whitening here. We report that control because the fitted AR model is not uniquely necessary for the observed improvement.

Our still-hand-versus-paper hypothesis was not established, so we withdrew it from the headline. These are exploratory results from one participant. Our contribution is verified preprocessing evidence with clear calibration boundaries, measurable responsiveness and reproducible failures. The next test needs an independent recording.
