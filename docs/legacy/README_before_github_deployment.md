# Whitening Under Test

A local ECoG signal lab testing how reliably—and how quickly—spectral whitening supports Rock / Scissors / Paper decoding across 90 trials. Whitening evidence is the central contribution; brain–hand history is retained as a secondary diagnostic.

**Established submission: 96.9% mean repeated 10-fold CV.** The historical 97.8% was one 88/90 run. Original data, submission code and historical results remain intact. Every new result is exploratory.

The updated `ECoG_G25_code` scripts reproduce **94.4% → 97.3%** with causal filter application. On identical CV seeds, **11 trials improve their average correctness fraction and none worsen**; the original one-sided sign-flip gives **p = 0.00044998**, rounded 0.0004. This uses 20 ten-fold seeds over 90 distinct trials. Its whole-record AR fit is retrospective, even though filter application is causal.

A source first-half calibration check gives **39/45 → 40/45** on later trials. A fixed first-difference control also reaches **97.3%**: fitted AR is a strong supported path, but is not uniquely necessary here. Separately, our pre-task-fit ablation gives **95.0% → 97.4%** at 2.25 seconds on matching repeated splits, and **82/90 → 88/90** at one second under purged chronological evaluation.

Still-hand versus cued Paper separation **was not established in follow-up** and is withdrawn from the main claim. It does not justify a movement-independent intention claim. No quantitative failed-replication score is invented. Earlier movement-history models remain an inconclusive secondary analysis.

## Run in this workspace

Python 3.13.5 and the pinned dependencies are tested. The existing `.venv` is ready:

```bash
cd /Users/sandyftkhaotung/Documents/GitHub/ecog-project
source .venv/bin/activate
python -m ecog_contribution run
streamlit run dashboard/app.py
```

For a new environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Open [the local dashboard](http://127.0.0.1:8501). Default recording: `/Users/sandyftkhaotung/Documents/GitHub/ecog-project/source/ECoG_Handpose.mat`. Default outputs: `/Users/sandyftkhaotung/Documents/GitHub/ecog-project/results/contribution_suite`. Optional CLI paths:

```bash
python -m ecog_contribution run --data /Users/sandyftkhaotung/Documents/GitHub/ecog-project/source/ECoG_Handpose.mat --output /Users/sandyftkhaotung/Documents/GitHub/ecog-project/results/contribution_suite
```

The dashboard reads the default output folder. Missing, incomplete, modified or stale caches show the regeneration command. Dashboard interactions never silently retrain. Pipeline runtime and processing timings are recorded in each run; sample waiting and glove reaction times remain separate. Computation uses one BLAS thread.

## Six views

- **Overview:** established submission, verified follow-up, calibration caveats and all 28 speed/channel settings.
- **Whitening evidence:** exact G25 reproduction; matched pre-task whitening on/off; first-half calibration; per-trial changes, uncertainty, spectra and first-difference/AR-order controls.
- **Reliability:** purged blocks, past-only expanding tests, held-out calibration, coverage, explicit abstentions and persistent channel loss after retraining.
- **Trial replay:** all 90 recorded trials, matched whitening on/off decoder selection, errors-only inspection, deadlines, play/pause and seeking. Default: pre-task AR, one second, 60 channels, purged blocks. Both predictions stay hidden until their samples are available.
- **Signal diagnostics:** cue/glove alignment, causal beta/high-gamma/low-frequency traces, unusual trials, and secondary history/baseline sensitivity.
- **Spatial signals:** numbered 10-by-6 layout, training-only selection stability, redundancy, referencing and fitted held-out LDA contributions.

The fixed 0.90 decide/wait policy accepts 77/90 trials, correctly classifies 75/77, and abstains on 13. Recorded replay shows real held-out mistakes. This is within-recording evaluation, not an independent participant test or deployed live BCI.

## Reproducible deliverables

`results/contribution_suite/` contains versioned metrics, CSV predictions and trial rows; NPZ splits, coefficients, probabilities, ablation/control predictions, display traces and explanatory arrays; and a manifest with code/data/source hashes and artifact checksums.

- [Contribution report](results/contribution_suite/CONTRIBUTION_REPORT.md)
- [140-second presentation guide](results/contribution_suite/PRESENTATION_140_SECONDS.md)
- [Unsent English/Korean team draft](results/contribution_suite/TEAM_UPDATE_DRAFT.md)
- `figures/06–09`: whitening evidence, calibration boundaries, spectrum and mechanism control, in PNG/SVG. `01–05` retain supporting brain–hand and speed/confidence figures.

Read [methods](docs/CONTRIBUTION_METHODS.md), [acceptance checks](docs/ACCEPTANCE.md), and [local sharing instructions](docs/SHARING_DASHBOARD.md). Refreshed portable archives and Discord-sized parts are in `shareable/`. No upload or team message was sent.

## Preserved work and interpretation

Historical `source/`, `whitening_compare.py`, `explore_ecog_angles.py`, `run_team_submission.py`, `team_submission_whole_record/`, historical results, and the supplied G25 scripts are unchanged. `python explore_ecog_angles.py` still runs the earlier experiment. Prior plans/drafts are in `docs/legacy/`; the prior contribution cache is archived in `results/archive/contribution_suite_movement_history_v1.zip`.

Glove values are diagnostic data, never classifier features or replacement labels. No trial is excluded or relabelled by quality flags. The recording supplies no diagnosis metadata; team notes describe epilepsy, so stroke-specific claims are unsupported. No clinical diagnosis, disease/hyperactivity correlation, exact cortical source or causal-connectivity inference is made. All new analyses concern one previously explored recording. Work remains local.
