import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import runpy,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
threadpool_limits(1)
root=Path(__file__).resolve().parents[1]
plt.show=lambda:plt.savefig(root/'outputs/temporal_generalization.png',dpi=180)
os.chdir(r'C:\Users\AbriK\Documents\ecog-hand-pose')
d=runpy.run_path('temporal_generalization.py')
out={}
for name,(M,chance) in d['results'].items():
    out[name]={'chance':chance,'make_make':float(d['block'](M,d['MAKE'],d['MAKE'])),'release_release':float(d['block'](M,d['RELEASE'],d['RELEASE'])),'make_release':float(d['block'](M,d['MAKE'],d['RELEASE'])),'release_make':float(d['block'](M,d['RELEASE'],d['MAKE']))}
with (root/'outputs/temporal_results.json').open('w') as f:json.dump(out,f,indent=2)
