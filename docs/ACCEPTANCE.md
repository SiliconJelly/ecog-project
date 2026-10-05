# Acceptance and verification

Run the suite, then `python -m pytest tests -q` in the workspace virtual environment.

Automated scientific tests cover retained-state batch/chunk equivalence, truncation and future modification, outer-test-label invariance of channel ranking/model/calibration, complete purged coverage, missing onset, absent training classes, nonfinite signals, flat calibration, missing/stale caches, all-abstain accounting and raw-channel omission before referencing. Floating-point filter equivalence uses numerical tolerances; identical chunk schedules are checked exactly.

Artifact tests recompute grid, confidence, nested-selection, forward, acquisition, dropout, montage and history metrics from CSV/NPZ exports; reconcile accepted/abstained/correct counts; verify 90 trial rows and 89 transitions; check temporal purging and past-only forward indices; and rehash protected original files. The CLI itself checks historical fold-score reproduction and actual-recording causal prefix equivalence before extending experiments.

Streamlit AppTest traverses all six pages, whitening comparisons and signal-diagnostic subviews, checks replay prediction gating, errors-only trials and empty electrode selection, and prohibits model fitting during interactions. Browser checks at desktop and narrow widths inspect dark rendering and exercise playback/seek controls. Native Streamlit labelled controls provide keyboard interaction; visible focus styling is retained. Responsive inspection does not assert a formal accessibility certification.

No prospective online BCI, independent participant, clinical interpretation or predetermined accuracy improvement is an acceptance requirement. The accepted outcome is reproducible evidence with mistakes, abstentions and negative findings exposed. The 140-second script is a rehearsal guide; an actual timed recording still requires the team's identity and speaker.

Completed verification in this workspace: all 19 acceptance tests passed; pinned dependency checks found no broken requirements; the final contribution cache verified successfully. Desktop (1440 × 1000) and narrow (640 × 900) browser inspection confirmed stacked controls, readable traces, prediction gating, error visibility and the numbered spatial view in the original layout check; the refreshed whitening view, first-half controls, paired replay and retained mistakes were also inspected. The full suite reproduced historical fold scores and preserved 16 protected inputs. See [layout and preview evidence](DESIGN.md).

Whitening-specific tests recompute source accuracy, helped/hurt trial fractions, the exact seeded one-sided sign-flip, AR-order and first-difference scores from exports; check all seven paired ablations and later-half/forward cohorts; and verify source hashes. Both replay decoder choices are tested with prediction gating and no fitting.

Protected reproduction-input changes now invalidate the cache and its UI fingerprint; a dedicated test checks modification and removal using temporary files. The portable manifest stores these paths relative to the project, preserving validation after extraction elsewhere.

Both refreshed ZIPs passed CRC checks. The smaller bundle was extracted into a different temporary directory with the exact recording supplied, and its cache plus all six dashboard views verified successfully. The 18 Discord parts and combine script reconstructed the full ZIP with an identical SHA-256.
