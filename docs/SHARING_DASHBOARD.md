# Share the dashboard for local use

The `shareable/` folder contains prepared archives. Nothing was uploaded or sent. The refreshed full ZIP is approximately 163.4 MiB; the existing-data ZIP is approximately 49.1 MiB. The fallback contains 18 numbered parts, each at most 10 MB, plus `combine_parts.py`.

- `ecog-dashboard-full.zip`: complete app, refreshed whitening analyses, recording, the three protected G25 reproduction scripts, historical reproduction inputs, pinned requirements and a portable launcher. This is the simplest team handoff.
- `ecog-dashboard-existing-data.zip`: the same bundle without the original recording. Use this if the team already has the exact ECoG_Handpose.mat file; place it in the extracted `source/` folder.
- `discord_parts/`: if the full ZIP exceeds your upload allowance, upload every numbered part and `combine_parts.py`. Download all into one directory and run `python combine_parts.py`, then extract the reconstructed ZIP.

The cache checks the recording, analysis code, protected reproduction inputs and artifact checksums. The bundle changes only its copy of the manifest recording path to `source/ECoG_Handpose.mat`. Computed results, checksums, and analysis source remain unchanged. `launch_dashboard.py` runs from the extracted project folder, avoiding dependence on the original user's absolute path.

Team setup instructions for macOS/Linux and Windows are in the archive's `README.md`. With Python 3.13 available:

```bash
cd ecog-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python launch_dashboard.py
```

Windows PowerShell:

```powershell
cd ecog-dashboard
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch_dashboard.py
```

Open http://127.0.0.1:8501 on the teammate's own computer. This address points to that computer's local server. No analysis rerun is necessary for the supplied cache. To regenerate bundles locally after analysis changes, run `python tools/package_dashboard.py` from this workspace.

## Updated whitening handoff

The refreshed bundles contain six dashboard views, exact G25 reproduction, paired whitening on/off exports, first-half calibration checks, first-difference/AR-order controls and updated scientific figures, report, presentation guide and unsent draft. Original 96.9% submission evidence remains separate. The still-hand/Paper claim is withdrawn. Replace older download parts together; the combine script verifies the current archive hash.
