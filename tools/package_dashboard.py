"""Build local team bundles. Does not upload or modify analysis artifacts."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'shareable'
PREFIX='ecog-dashboard/'
TEAM_README='''# Whitening Under Test — team local setup

Use Python 3.13. Extract the ZIP and open a terminal in the ecog-dashboard folder.

For the full bundle the recording is included. For the lighter bundle, copy your existing
ECoG_Handpose.mat to source/ECoG_Handpose.mat. Its SHA-256 must match:
b2b65f0f040ee9663c523801d92ba7e685d2ade94a2c1592c9975320966fab60

macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python launch_dashboard.py
```

Windows PowerShell:
```
py -3.13 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe launch_dashboard.py
```

Open http://127.0.0.1:8501 in your own browser. No model training is needed to view the cached results.
The launcher sets the working directory so the portable recording path resolves correctly.
All analysis code, cached results, scientific figures and the historical reproduction inputs are included.
To regenerate after changing analysis code: python -m ecog_contribution run

Six views show whitening evidence, overview, reliability, recorded-data replay,
signal diagnostics and spatial signals. Replay compares matched whitening on/off models.
The G25 causal comparison reproduces 94.4% -> 97.3%, 11 improved trial fractions,
none worsened, one-sided p=0.00044998. Its whole-record AR fit is retrospective.
The source first-half holdout is 39/45 -> 40/45; a first-difference control ties 97.3%.
Pre-task calibration and purged comparisons are separately labelled. Still-hand/Paper
separation is not established and has been withdrawn from the main claim.
The original 96.9% submission and every exploratory result remain distinguished.
Reports, the presentation guide and the unsent team draft are under results/contribution_suite/.
No environment folder is distributed; dependencies install locally. Tested Python 3.13.5,
with the package versions pinned in requirements.txt.
'''
LAUNCHER='''from pathlib import Path
import os
import subprocess
import sys

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]))
'''


def build():
    OUT.mkdir(exist_ok=True)
    from ecog_contribution.artifacts import validate_cache
    ok,message=validate_cache()
    if not ok:raise ValueError(message)
    files=set()
    for folder,pattern in [('ecog_contribution','*.py'),('dashboard','*.py'),('source','*.py'),('results/contribution_suite','*')]:
        files.update(p for p in (ROOT/folder).rglob(pattern) if p.is_file() and '__pycache__' not in p.parts)
    names=['requirements.txt','.streamlit/config.toml','whitening_compare.py','explore_ecog_angles.py','run_team_submission.py',
           'team_submission_whole_record/classify_whitened.py','team_submission_whole_record/METHOD_NOTES.md',
           'team_submission_whole_record/results/on/arrays.npz','team_submission_whole_record/results/off/arrays.npz',
           'results/whitening/comparison_arrays.npz','docs/CONTRIBUTION_METHODS.md','docs/DESIGN.md',
           'docs/SHARING_DASHBOARD.md','docs/ACCEPTANCE.md',
           'ECoG_G25_code/3_oct5_analyses/common.py','ECoG_G25_code/4_partner_checks/partner_checks.py',
           'ECoG_G25_code/4_partner_checks/whitening_audit.py']
    files.update(ROOT/n for n in names)
    files.update(p for p in (ROOT/'docs/images').glob('*.png') if p.is_file())
    for include_data,name in [(True,'ecog-dashboard-full.zip'),(False,'ecog-dashboard-existing-data.zip')]:
        with zipfile.ZipFile(OUT/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in sorted(files):
                relative=str(p.relative_to(ROOT))
                if p.name=='manifest.json':
                    manifest=json.loads(p.read_text());manifest['data_path']='source/ECoG_Handpose.mat'
                    archive.writestr(PREFIX+relative,json.dumps(manifest,indent=2)+'\n')
                else:archive.write(p,PREFIX+relative)
            if include_data:archive.write(ROOT/'source/ECoG_Handpose.mat',PREFIX+'source/ECoG_Handpose.mat')
            archive.writestr(PREFIX+'README.md',TEAM_README)
            archive.writestr(PREFIX+'launch_dashboard.py',LAUNCHER)
        print(f'{name}: {(OUT/name).stat().st_size/1024**2:.1f} MiB')
    full=OUT/'ecog-dashboard-full.zip';parts=OUT/'discord_parts';parts.mkdir(exist_ok=True)
    digests=[]
    with full.open('rb') as recording:
        while block:=recording.read(10_000_000):
            name=f'ecog-dashboard-full.zip.part{len(digests)+1:03d}'
            (parts/name).write_bytes(block);digests.append(hashlib.sha256(block).hexdigest())
    digest=hashlib.sha256(full.read_bytes()).hexdigest()
    combine=f'''# Download every part and this script into the same folder, then run python combine_parts.py.
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parent
expected={digests!r}
output=root/"ecog-dashboard-full.zip"
for i,digest in enumerate(expected,1):
    part=root/f"ecog-dashboard-full.zip.part{{i:03d}}"
    if not part.exists() or hashlib.sha256(part.read_bytes()).hexdigest()!=digest:
        raise SystemExit(f"Missing or damaged part: {{part.name}}")
with output.open("wb") as target:
    for i in range(1,len(expected)+1):
        target.write((root/f"ecog-dashboard-full.zip.part{{i:03d}}").read_bytes())
assert hashlib.sha256(output.read_bytes()).hexdigest()=={digest!r}
print("Verified ZIP created. Extract ecog-dashboard-full.zip and follow README.md inside.")
'''
    (parts/'combine_parts.py').write_text(combine)
    for old in parts.glob('ecog-dashboard-full.zip.part*'):
        if int(old.name.rsplit('part',1)[1])>len(digests):old.unlink()
    print(f'Discord fallback: {len(digests)} parts, each at most 10 MB, plus combine_parts.py')


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    build()
