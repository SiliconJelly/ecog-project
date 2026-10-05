# Team G25: ECoG hand-pose code

All analyses from Oct 4-5, 2026. The summary of results is in the team doc "Team G25: ECoG Analyses Run and Checked".

## Setup

1. Put `ECoG_Handpose.mat` (118 MB, not included) in the folder you run from.
2. `pip install numpy scipy scikit-learn matplotlib threadpoolctl`
3. For `3_oct5_analyses` and `4_partner_checks`, copy `3_oct5_analyses/common.py` next to the script you run, and create `results/` and `figures/` folders there. Every analysis imports `common.py`.
4. Set `OMP_NUM_THREADS=1` for faster permutation tests on laptops.

## Folders

| Folder | What it is |
|---|---|
| `1_original_scripts` | Abri's first scripts (classify, high_gamma, baseline_state, carryover, fist_vs_peace_residue, effort_state, local_vs_coupling, temporal_generalization, look, join_mat). They display figures with `plt.show()`. To save PNGs, run them through `3_oct5_analyses/runwrap.py`. |
| `2_oct4_runners` | The Oct 4 reproduction (decoder, history transfer, blocked holdout, phase transfer). Edit the data path at the top. |
| `3_oct5_analyses` | All Oct 5 tests (list below). |
| `4_partner_checks` | Checks of Bagus's scripts: `partner_checks.py` (whitening + predictive), `whitening_audit.py` (deck numbers, mechanism), `paradox_check.py` (pose paradox; it expects his `pose_state_paradox.py` in his folder, so edit the path in `os.chdir`). |
| `figures` | Every figure used in the report. |

## 3_oct5_analyses

| Script | Test | Runtime |
|---|---|---|
| `common.py` | Shared loading and preprocessing; `causal=True` = forward-only filters | — |
| `causal_rerun.py` | Pre-cue beta, history decoding, carryover: zero-phase vs causal vs guard window | ~10 min |
| `finger_context.py`, `finger_extra.py` | Posture vs neural pre-cue state; gestures as finger combinations; errors vs posture | ~5 min |
| `phase_permutation.py [n n]` | Closing vs opening transfer, label + phase-swap permutation | ~60 min at 500 |
| `glove_aligned_phase.py [n]`, `glove_label_perm.py [n]` | Same, aligned to glove onsets | ~30 min |
| `pfix_phase.py <job> <n>`, `pfix_decode.py [n]` | P-value recheck with matched CV repeats | ~20 min/job |
| `release_window.py`, `combined_decoder.py`, `fusion_checks.py`, `window_nested.py` | Release window, late fusion, paired tests, accuracy vs delay, nested window choice | ~5-60 min |
| `robust_time.py`, `robust_control.py` | Predicting only future trials; training-size control | ~5 min |
| `onset_locked.py` | Features locked to brain-detected onset | ~3 min |
| `shared_onset.py`, `shared_perm.py` | Channels shared by closing and opening; permutation with in-fold selection | ~10 min |
| `finger_decoder.py` | Continuous finger decoding; leave-one-gesture-out | ~5 min |
| `same_hand.py`, `same_hand_sens.py` | Deliberate paper vs rest, posture-matched | ~10 min |
| `tg_transition.py [n]` | Temporal generalization, rest -> transition -> movement | ~25 min at 100 |
| `transition_relationship.py` | How the signal builds per gesture | ~2 min |
| `timing_variance.py`, `state_increment.py` | Onset detector (held out, rest false alarms); variance explained by pre-cue state | ~10 min |
| `runwrap.py <script.py>` | Runs any original script and saves its figures as PNGs | — |
