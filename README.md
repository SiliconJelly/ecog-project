# Whitening Under Test 🧠

A Streamlit dashboard exploring how spectral whitening improves ECoG Rock / Scissors / Paper decoding. Browse whitening comparisons, held-out predictions, trial replay, reliability, and signal diagnostics across 90 trials.

## Run locally

Use Python 3.13 and [Git LFS](https://git-lfs.com/) for the recording:

```bash
git lfs install
git clone https://github.com/SiliconJelly/ecog-project.git
cd ecog-project
git lfs pull
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Verified results are included; no training is needed. To regenerate them, run `python -m ecog_contribution run` from the repository root.

## Deploy

On [Streamlit Community Cloud](https://share.streamlit.io/), select `SiliconJelly/ecog-project`, branch `main`, entrypoint `dashboard/app.py`, and **Python 3.13** under Advanced settings. Dependencies come from `requirements.txt`; [Community Cloud supports Git LFS](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Research & license

These exploratory results come from one recording; the dashboard replays held-out predictions. See [methods](docs/CONTRIBUTION_METHODS.md) and the [contribution report](results/contribution_suite/CONTRIBUTION_REPORT.md) for evidence and limitations.

Project code is [MIT licensed](LICENSE). Supplied recordings and third-party materials retain their original terms.
