# Contribution direction: Whitening Under Test

Updated 6 October 2026. The central question is **whether spectral whitening improves gesture decoding when the remaining processing and evaluation stay fixed**. Speed, confidence, temporal calibration and channel robustness support that argument. Brain–hand history remains a diagnostic appendix.

The supplied G25 comparison reproduces 94.4% → 97.3%, 11 improved trial correctness fractions and none worsened, one-sided p=0.00044998. The whole-record coefficient fit is retrospective despite causal filtering. The source later-half check gives 39/45 → 40/45; a fitted-free first difference ties AR at 97.3%. Keep these results visible rather than claiming fitted AR is uniquely necessary or that the headline proves prospective accuracy.

The contribution suite adds matched pre-task whitening ablations at seven deadlines, purged chronological blocks, a purged midpoint holdout and past-only expanding tests. Confidence, trial replay and lost-channel retraining remain concrete engineering checks. The established 96.9% submission remains unchanged and separately labelled.

Still-hand versus cued Paper separation was not established in follow-up and is withdrawn from the central narrative. Earlier history-adjusted neural models were independently inconclusive. Neither supports a movement-independent intention or clinical claim.

Current measured evidence, figures and interpretations: [contribution report](results/contribution_suite/CONTRIBUTION_REPORT.md). The [140-second guide](results/contribution_suite/PRESENTATION_140_SECONDS.md) centers matched whitening, calibration boundaries, the mechanism control and visible failures. The [team draft](TEAM_UPDATE.md) is unsent. Prior plans and drafts are preserved in `docs/legacy/` and previous cache outputs in `results/archive/`.

Run `python -m ecog_contribution run`, then `streamlit run dashboard/app.py`. Installation, actual workspace paths and portable team bundles are documented in [README](README.md). Work stays local; no publishing, upload, messaging, Git initialization or remote edit is included.
