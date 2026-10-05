"""
Nested check (Oct 5): the closing-only decoder looked best when stopped at 2.0 s (97.1%),
but that end time was picked by looking at CV accuracy. Here the end time is chosen
INSIDE each training fold (inner 5-fold CV over 1.0-2.25 s), then scored on the outer
held-out trials. Also a paired test: stop-at-2.0 s vs the original 2.25 s, identical folds.
Causal preprocessing.
"""
import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
import common as C
rng = np.random.default_rng(23)
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(50, 300))
lda = lambda: LDA(solver="lsqr", shrinkage="auto")
ENDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
X = {T: C.decoder_features(hg, on, np.arange(0.25, T + 1e-9, 0.25)) for T in ENDS}
acc, chosen = [], []
c20, c225 = np.zeros(90), np.zeros(90)
for s in range(10):
    pred = np.zeros(90, int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(X[2.25], labels):
        inner = {T: cross_val_score(lda(), X[T][tr], labels[tr], cv=StratifiedKFold(5, shuffle=True, random_state=100 + s)).mean() for T in ENDS}
        T = max(ENDS, key=lambda t: (inner[t], -t))
        chosen.append(T)
        pred[te] = lda().fit(X[T][tr], labels[tr]).predict(X[T][te])
        c20[te] += (lda().fit(X[2.0][tr], labels[tr]).predict(X[2.0][te]) == labels[te]) / 10
        c225[te] += (lda().fit(X[2.25][tr], labels[tr]).predict(X[2.25][te]) == labels[te]) / 10
    acc.append(np.mean(pred == labels))
d = c20 - c225
null = np.array([(d * rng.choice([-1, 1], 90)).mean() for _ in range(20000)])
p = float((np.sum(null >= d.mean()) + 1) / 20001)
vals, cnt = np.unique(chosen, return_counts=True)
print(f"Nested (end time chosen inside training folds): {np.mean(acc)*100:.1f}%")
print("  end times chosen:", {float(v): int(c) for v, c in zip(vals, cnt)})
print(f"Paired, stop at 2.0 s vs original 2.25 s: {c20.mean()*100:.1f}% vs {c225.mean()*100:.1f}%; helped {np.sum(d>0)}, hurt {np.sum(d<0)}; sign-flip p={p:.4f}")
json.dump(dict(nested_acc=float(np.mean(acc)), chosen={str(float(v)): int(c) for v, c in zip(vals, cnt)},
               stop20=float(c20.mean()), stop225=float(c225.mean()), p_signflip=p,
               helped=(np.where(d > 0)[0] + 1).tolist(), hurt=(np.where(d < 0)[0] + 1).tolist()),
          open("results/window_nested.json", "w"), indent=2)
