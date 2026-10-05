"""
Follow-up (Oct 5): WHERE is the shared fist/peace information between closing and opening?

Glove-aligned (as in Test 4), fist vs peace, causal high-gamma, 0-0.7 s after each onset.
1. Per-channel fist-minus-peace effect (t) during closing and during opening.
   Channels with the SAME sign in both phases carry shared information; opposite sign = flipped.
2. Per-channel cross-phase transfer: single-channel LDA trained on closing, tested on opening
   (and reverse), 5-fold x 5 repeats, averaged over the window.
3. Transfer using only the top-k 'shared' channels vs all 60 (channels chosen INSIDE each
   training fold, so no leakage).
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import sys

src = open("glove_aligned_phase.py").read().split("def blocks(")[0]
g = {"__name__": "gp"}
sys.argv = ["x", "1"]
exec(compile(src, "gp_head", "exec"), g)
Xm, Xr, yy, W = g["Xm"], g["Xr"], g["yy"], g["W"]
M = Xm[:, :, W].mean(2)          # trials x 60, closing window mean
R = Xr[:, :, W].mean(2)          # opening window mean
f, p_ = yy == 1, yy == 2

tM = stats.ttest_ind(M[f], M[p_]).statistic
tR = stats.ttest_ind(R[f], R[p_]).statistic
rr = stats.pearsonr(tM, tR)
print(f"Channel effect maps (fist - peace): correlation between closing and opening r={rr.statistic:+.2f} (p={rr.pvalue:.3f})")
shared = np.where((np.sign(tM) == np.sign(tR)) & (np.abs(tM) > 2) & (np.abs(tR) > 2))[0]
flipped = np.where((np.sign(tM) != np.sign(tR)) & (np.abs(tM) > 2) & (np.abs(tR) > 2))[0]
print(f"  same sign, |t|>2 in both: CH {(shared+1).tolist()}")
print(f"  opposite sign, |t|>2 in both: CH {(flipped+1).tolist()}")
print(f"  |t|>2 closing only: {int(np.sum((np.abs(tM)>2)&(np.abs(tR)<=2)))} ch; opening only: {int(np.sum((np.abs(tR)>2)&(np.abs(tM)<=2)))} ch")

# per-channel transfer
def chan_transfer(A, B):
    acc = np.zeros(60)
    for s in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(A, yy):
            for c in range(60):
                m = LDA().fit(A[tr][:, [c]], yy[tr])
                acc[c] += np.sum(m.predict(B[te][:, [c]]) == yy[te])
    return acc / (5 * len(yy))
tr_mr, tr_rm = chan_transfer(M, R), chan_transfer(R, M)
tr_avg = (tr_mr + tr_rm) / 2
best = np.argsort(tr_avg)[::-1][:8]
print("  best single-channel cross-phase transfer:", ", ".join(f"CH{c+1} {tr_avg[c]*100:.0f}%" for c in best))

# top-k shared channels chosen inside each fold
def subset_transfer(k, repeats=10):
    res = []
    for s in range(repeats):
        a = 0
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(M, yy):
            t1 = stats.ttest_ind(M[tr][yy[tr] == 1], M[tr][yy[tr] == 2]).statistic
            t2 = stats.ttest_ind(R[tr][yy[tr] == 1], R[tr][yy[tr] == 2]).statistic
            score = np.where(np.sign(t1) == np.sign(t2), np.minimum(abs(t1), abs(t2)), 0)
            ch = np.argsort(score)[::-1][:k] if k < 60 else np.arange(60)
            m1 = LDA(solver="lsqr", shrinkage="auto").fit(M[tr][:, ch], yy[tr])
            m2 = LDA(solver="lsqr", shrinkage="auto").fit(R[tr][:, ch], yy[tr])
            a += np.sum(m1.predict(R[te][:, ch]) == yy[te]) + np.sum(m2.predict(M[te][:, ch]) == yy[te])
        res.append(a / (2 * len(yy)))
    return float(np.mean(res))
sub = {k: subset_transfer(k) for k in (3, 5, 10, 20, 60)}
print("  cross-phase transfer with top-k shared channels (chosen in training folds):",
      ", ".join(f"k={k}: {v*100:.1f}%" for k, v in sub.items()))
json.dump(dict(map_corr=float(rr.statistic), map_corr_p=float(rr.pvalue), shared=(shared + 1).tolist(),
               flipped=(flipped + 1).tolist(), best_channels={f"CH{c+1}": float(tr_avg[c]) for c in best},
               subset_transfer={str(k): v for k, v in sub.items()}), open("results/shared_onset.json", "w"), indent=2)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
norm = TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6)
for a, t, title in [(ax[0], tM, "Closing: fist − peace (t)"), (ax[1], tR, "Opening: fist − peace (t)")]:
    im = a.imshow(t.reshape(6, 10).T, cmap="RdBu_r", norm=norm)
    for ch in range(60):
        a.text(ch // 10, ch % 10, ch + 1, ha="center", va="center", fontsize=7)
    a.set(title=title, xticks=[], yticks=[])
fig.colorbar(im, ax=ax[:2], shrink=.8, label="t (red = more in fist)")
im2 = ax[2].imshow(tr_avg.reshape(6, 10).T * 100, cmap="Purples", vmin=45, vmax=65)
for ch in range(60):
    ax[2].text(ch // 10, ch % 10, ch + 1, ha="center", va="center", fontsize=7)
ax[2].set(title="Single-channel cross-phase\ntransfer (%)", xticks=[], yticks=[])
fig.colorbar(im2, ax=ax[2], shrink=.8)
ax[3].scatter(tM, tR, s=25, color="#2a78d6")
for c in np.r_[shared, flipped]:
    ax[3].annotate(f"CH{c+1}", (tM[c], tR[c]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[3].axhline(0, color="k", lw=.5); ax[3].axvline(0, color="k", lw=.5)
ax[3].set(xlabel="t during closing", ylabel="t during opening", title=f"Same channels, same direction?\nr={rr.statistic:+.2f}")
ax[3].spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D4_shared_onset.png", dpi=160)
