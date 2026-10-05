# Same Gesture, Different Starting State

A local brain–hand signal lab for the ECoG Rock / Scissors / Paper case study. The contribution connects decoding speed, calibrated reliability, hand behavior, and spatial signals across 90 trials from one recording.

**Established submission: 96.9% mean accuracy across repeated 10-fold cross-validation.** The historical 97.8% was one 88/90 run. The submission uses whole-record AR whitening; every new causal, temporal, history, and robustness result is separately labelled exploratory. This suite does not replace the submission or claim a new generalization result.

The new fixed one-second, 60-channel causal decoder gives **88/90 correct in five purged chronological blocks**. The fixed 0.90-confidence decide/wait policy accepts **77/90**, correctly decoding **75/77**, and abstains on **13**. The primary history analysis finds an adjusted absolute high-gamma association of approximately **−0.047 dB per SD of preceding movement**, with a within-recording interval **[−0.128, 0.052]**. Adding history does not improve chronological prediction after hand-behavior adjustment. These measured negative results are part of the contribution.

## Run in this workspace

Use Python 3.13 (tested 3.13.5), from the actual project directory:

```bash
cd /Users/sandyftkhaotung/Documents/GitHub/ecog-project
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ecog_contribution run
streamlit run dashboard/app.py
```

The existing `.venv` is ready. Open [the local dashboard](http://127.0.0.1:8501). The complete suite takes roughly four to five minutes on this machine; processing timings are recorded for each run. Model computation uses one BLAS thread for repeatability. Dependencies are pinned to the tested versions in `requirements.txt`; pytest is in `requirements-dev.txt`.

Default input: `/Users/sandyftkhaotung/Documents/GitHub/ecog-project/source/ECoG_Handpose.mat`.
Default output: `/Users/sandyftkhaotung/Documents/GitHub/ecog-project/results/contribution_suite`.

```bash
python -m ecog_contribution run --data /Users/sandyftkhaotung/Documents/GitHub/ecog-project/source/ECoG_Handpose.mat --output /Users/sandyftkhaotung/Documents/GitHub/ecog-project/results/contribution_suite
python -m pytest tests -q
```

Custom output folders are supported by the CLI; the dashboard deliberately reads the default contribution-suite folder. Missing, incomplete, modified or stale results display a regeneration command. Interactions read cached artifacts and never fit a model. Keep the recording locally available for provenance verification.

## Five views

- **Overview:** historical versus exploratory evidence, all 28 speed/channel combinations and evaluation protocols.
- **Reliability:** purged temporal blocks, past-only forward tests, inner-fold temperature calibration, coverage, abstentions and persistent channel loss after retraining.
- **Trial replay:** all 90 recorded trials, errors-only inspection, deadline selection, play/pause, seeking and finger/neural traces. The default is one second / 60 channels / purged chronological evaluation. Predictions remain hidden until their input deadline.
- **Neuroscience:** movement-controlled history models, baseline sensitivity, cue/glove alignment, causal beta/high-gamma/low-frequency responses and unusual trials.
- **Spatial signals:** documented numbered 10-by-6 layout, training-only selection frequency, feature redundancy, common-average/local-neighbor referencing and held-out fitted score contributions.

## Deliverables

All new artifacts live in `results/contribution_suite/`:

- `metrics.json`, `predictions.csv`, `trial_table.csv`: metrics, held-out predictions and one row per trial.
- `model_arrays.npz`: feature arrays, split indices, predictions, calibrated probabilities and contributions; temperature parameters are in `metrics.json`.
- `display_traces.npz`, `history_arrays.npz`: processed display traces, alignments and held-out explanatory predictions.
- `manifest.json`: schema/config/code/data hashes and checksums of all outputs.
- [Contribution report](results/contribution_suite/CONTRIBUTION_REPORT.md), [140-second presentation guide](results/contribution_suite/PRESENTATION_140_SECONDS.md), [unsent team-update draft](results/contribution_suite/TEAM_UPDATE_DRAFT.md).
- `figures/`: PNG and SVG scientific figures: annotated trial through the next cue, same gesture/different movement, adjusted history effect, two unusual trials, speed/confidence.

Read [methods and interpretation](docs/CONTRIBUTION_METHODS.md) for exact window, referencing, leakage-control and uncertainty definitions. [Acceptance checks](docs/ACCEPTANCE.md) records what is tested and what the replay can demonstrate.

## Existing work stays available

`source/`, `whitening_compare.py`, `run_team_submission.py`, `explore_ecog_angles.py`, historical `results/whitening/` and `results/contribution_angles/`, and `team_submission_whole_record/` remain intact. The earlier exploratory command is still `python explore_ecog_angles.py`. The original README is preserved in [docs/legacy/README_before_contribution_suite.md](docs/legacy/README_before_contribution_suite.md).

The matrix supplies no diagnosis metadata. Team notes describe an epilepsy participant; a stroke-specific interpretation is unsupported. No clinical diagnosis, seizure localization, disease/hyperactivity claim, precise source localization or causal connectivity inference is made. Glove values are diagnostic/explanatory data and never decoder inputs or substitute labels. No trial is excluded or relabelled by quality or unusual-behavior flags. Repeated windows and trials are repeated measurements from one participant.

Work is local: no remote repository changes, uploading, publishing or team messaging.

## Share with the team

Prepared portable archives are in `shareable/`. Use [sharing instructions](docs/SHARING_DASHBOARD.md) for the complete ZIP, a smaller archive for teammates who already have the recording, and numbered Discord upload parts. Archives include a portable launcher and preserve the verified analysis outputs. Nothing has been uploaded or sent.
