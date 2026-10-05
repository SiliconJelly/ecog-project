"""
P-value check (Oct 5) for the pre-cue decoding tests in causal_rerun.py and
finger_context.py: the shuffled null now averages the SAME number of CV seeds
as the real score (3 for causal_rerun, 5 for finger_context). Causal filtering.
"""
import json, sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score
import common as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(21)
fs = C.fs
y, cue, glove, on, off, labels = C.load_raw()
prev = np.r_[0, labels[:-1]]
hp = np.arange(90) > 0
P = {b: C.band_power(b, True) for b in C.BANDS}
Nf = np.array([np.concatenate([C.win_log(P[b], o, -1, 0) for b in C.BANDS]) for o in on])
pre_post = np.array([glove[:, o - fs:o].mean(1) for o in on])
Z = np.c_[pre_post, np.arange(90) / 90]


def bal(F, yy, seeds, res=None):
    accs = []
    for s in seeds:
        pred = np.zeros(len(yy), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(F, yy):
            Ftr, Fte = F[tr], F[te]
            if res is not None:
                reg = LinearRegression().fit(res[tr], Ftr)
                Ftr, Fte = Ftr - reg.predict(res[tr]), Fte - reg.predict(res[te])
            pred[te] = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(Ftr, yy[tr]).predict(Fte)
        accs.append(balanced_accuracy_score(yy, pred))
    return float(np.mean(accs))


fp = (prev == 1) | (prev == 2)
tests = [
    ("causal_rerun B: previous moved, brain", Nf[hp], (prev[hp] != 3).astype(int), None, range(3)),
    ("causal_rerun C: previous fist vs peace, brain", Nf[fp], prev[fp], None, range(3)),
    ("finger_context: previous moved, brain only", Nf[hp], (prev[hp] != 3).astype(int), None, range(5)),
    ("finger_context: previous moved, glove only", pre_post[hp], (prev[hp] != 3).astype(int), None, range(5)),
    ("finger_context: previous moved, brain + glove", np.c_[Nf, pre_post][hp], (prev[hp] != 3).astype(int), None, range(5)),
    ("finger_context: previous moved, brain with glove removed", Nf[hp], (prev[hp] != 3).astype(int), Z[hp], range(5)),
]
out = {}
for name, F, yy, res, seeds in tests:
    real = bal(F, yy, seeds, res)
    null = np.array([bal(F, rng.permutation(yy), seeds, res) for _ in range(N)])
    p = float((np.sum(null >= real) + 1) / (N + 1))
    out[name] = dict(acc=real, null95=float(np.percentile(null, 95)), p=p, seeds=len(seeds), n_perm=N)
    print(f"  {name:58s} {real*100:5.1f}%  shuffled 95th {np.percentile(null,95)*100:4.1f}%  p={p:.3f}", flush=True)
json.dump(out, open("results/pfix_decode.json", "w"), indent=2)
