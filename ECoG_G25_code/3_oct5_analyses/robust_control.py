"""
Control for robust_time.py: is the drop for the 'fixed' decoder caused by TIME
(drift) or just by training on only 30 trials?
Same training size, but the 30 training trials are drawn at random from the whole
session (10 per gesture), tested on the remaining 60. 200 draws.
Also size-matched controls for 'expanding' (train n random earlier-or-later trials).
"""
import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import common as C
y, cue, glove, on, off, labels = C.load_raw()
rng = np.random.default_rng(5)
out = {}
for causal in (True, False):
    tag = "causal" if causal else "zero-phase"
    X = C.decoder_features(C.band_power(None, causal, lo_hi=(50, 300)), on)
    # time-ordered first 30: class counts
    counts = [int(np.sum(labels[:30] == k)) for k in (1, 2, 3)]
    accs = []
    for _ in range(200):
        tr = np.concatenate([rng.choice(np.where(labels == k)[0], c, replace=False) for k, c in zip((1, 2, 3), counts)])
        te = np.setdiff1d(np.arange(90), tr)
        m = LDA(solver="lsqr", shrinkage="auto").fit(X[tr], labels[tr])
        accs.append(np.mean(m.predict(X[te]) == labels[te]))
    accs = np.array(accs)
    # where does the real 'fixed' (first 30 -> last 60) fall?
    m = LDA(solver="lsqr", shrinkage="auto").fit(X[:30], labels[:30])
    fixed = float(np.mean(m.predict(X[30:]) == labels[30:]))
    p = float((np.sum(accs <= fixed) + 1) / (len(accs) + 1))
    # expanding, size matched: for each block start s, train on s random trials excluding the test block
    exp_c = []
    for _ in range(50):
        c = []
        for s in range(30, 90, 10):
            te = np.arange(s, min(s + 10, 90))
            pool = np.setdiff1d(np.arange(90), te)
            tr = rng.choice(pool, s, replace=False)
            m = LDA(solver="lsqr", shrinkage="auto").fit(X[tr], labels[tr])
            c.extend(m.predict(X[te]) == labels[te])
        exp_c.append(np.mean(c))
    out[tag] = dict(class_counts_first30=counts, fixed_time_ordered=fixed, random30_mean=float(accs.mean()),
                    random30_5_95=np.percentile(accs, [5, 95]).tolist(), p_time_worse_than_random=p,
                    expanding_size_matched_random_mean=float(np.mean(exp_c)))
    print(f"[{tag}] first-30 class counts {counts}")
    print(f"  fixed, time-ordered (1-30 -> 31-90): {fixed*100:.1f}%")
    print(f"  same size, random trials:            {accs.mean()*100:.1f}%  (5-95%: {np.percentile(accs,5)*100:.0f}-{np.percentile(accs,95)*100:.0f}%)  p(time-ordered this low)={p:.3f}")
    print(f"  expanding, size-matched random:      {np.mean(exp_c)*100:.1f}%", flush=True)
json.dump(out, open("results/robust_control.json", "w"), indent=2)
