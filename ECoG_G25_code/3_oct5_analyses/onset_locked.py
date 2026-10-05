"""
Decoder test C (Oct 5): lock features to a movement onset detected from the BRAIN (no glove).

1. Detect onset per trial from motor high-gamma (causal): first time after the cue the
   smoothed signal on the 5 most active channels exceeds pre-cue mean + 3 SD for 100 ms.
   No crossing (most open trials barely move) -> use the median onset instead.
   Check against the glove onset for fist/peace trials.
2. Decoder features: same 8 x 250 ms bins as the original, but starting at
   (onset - 0.25 s) instead of cue + 0.25 s (same 2 s span).
3. Compare with the cue-locked decoder on identical CV splits.
"""
import json
import numpy as np
import scipy.signal as ss
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import common as C

fs = C.fs
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(50, 300))
TOP = [25, 36, 26, 15, 46]

# --- neural onset ---
env = ss.lfilter(np.ones(int(.05 * fs)) / int(.05 * fs), 1, hg[TOP].mean(0))   # causal 50 ms moving average
logenv = np.log10(env + 1e-12)
neu_on = np.full(90, np.nan)
for i, o in enumerate(on):
    base = logenv[o - fs:o]
    thr = base.mean() + 3 * base.std()
    seg = logenv[o:o + int(1.5 * fs)] > thr
    run = np.convolve(seg, np.ones(int(.1 * fs)), "valid") >= int(.1 * fs)
    idx = np.where(run)[0]
    if len(idx):
        neu_on[i] = idx[0] / fs
detected = np.isfinite(neu_on)
med = np.nanmedian(neu_on[labels != 3])
onset = np.where(detected, neu_on, med)
print("Neural onset detected:", {n: f"{np.sum(detected & (labels == k))}/30" for k, n in [(1, 'fist'), (2, 'peace'), (3, 'open')]})
print(f"  median neural onset (fist/peace) {med:.3f} s after cue")

# glove onset for comparison (same rule as glove_aligned_phase.py)
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)
glove_on = np.full(90, np.nan)
for i, o in enumerate(on):
    base = gs[:, o - fs:o].mean(1)
    d = np.sqrt(((gs[:, o:o + 2 * fs] - base[:, None]) ** 2).mean(0))
    h = np.where(d > max(.02, .3 * np.percentile(d, 95)))[0]
    if len(h):
        glove_on[i] = h[0] / fs
m = (labels != 3) & detected & np.isfinite(glove_on)
r = stats.spearmanr(neu_on[m], glove_on[m])
lead = np.median(glove_on[m] - neu_on[m])
print(f"  fist/peace: neural vs glove onset rho={r.statistic:.2f} (p={r.pvalue:.3g}), brain leads hand by median {lead*1000:.0f} ms")

# --- decoders ---
E = np.arange(0.25, 2.26, 0.25)


def feats(starts):
    return np.array([np.concatenate([C.win_log(hg, o, s0 + a - 0.25, s0 + b - 0.25) for a, b in zip(E[:-1], E[1:])])
                     for o, s0 in zip(on, starts)])


X_cue = feats(np.full(90, 0.25))        # identical to original: 0.25-2.25 s after cue
X_on = feats(onset)                     # onset-0.25 ... onset+1.75


def cv(X, runs=20):
    accs, CM = [], np.zeros((3, 3))
    for s in range(runs):
        pred = np.zeros(90, int)
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(X, labels):
            pred[te] = LDA(solver="lsqr", shrinkage="auto").fit(X[tr], labels[tr]).predict(X[te])
        accs.append(np.mean(pred == labels)); CM += confusion_matrix(labels, pred)
    return np.array(accs), CM / runs


a_cue, CM_cue = cv(X_cue)
a_on, CM_on = cv(X_on)
d = a_on - a_cue
w = stats.wilcoxon(a_on, a_cue) if np.any(d != 0) else None
out = dict(detected={n: int(np.sum(detected & (labels == k))) for k, n in [(1, 'fist'), (2, 'peace'), (3, 'open')]},
           median_onset=float(med), onset_vs_glove_rho=float(r.statistic), brain_leads_ms=float(lead * 1000),
           cue_locked=dict(acc=float(a_cue.mean()), errors=float(CM_cue.sum() - np.trace(CM_cue)), fist_peace=float(CM_cue[0, 1] + CM_cue[1, 0])),
           onset_locked=dict(acc=float(a_on.mean()), errors=float(CM_on.sum() - np.trace(CM_on)), fist_peace=float(CM_on[0, 1] + CM_on[1, 0])),
           runs_better=int(np.sum(d > 0)), runs_worse=int(np.sum(d < 0)), wilcoxon_p=None if w is None else float(w.pvalue))
print(f"  cue-locked   {a_cue.mean()*100:.1f}%  errors {out['cue_locked']['errors']:.2f} (fist<->peace {out['cue_locked']['fist_peace']:.2f})")
print(f"  onset-locked {a_on.mean()*100:.1f}%  errors {out['onset_locked']['errors']:.2f} (fist<->peace {out['onset_locked']['fist_peace']:.2f})")
print(f"  runs better {out['runs_better']}/20, worse {out['runs_worse']}/20" + ("" if w is None else f", Wilcoxon p={w.pvalue:.3f} (runs share data: descriptive only)"))
json.dump(out, open("results/onset_locked.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
for k, col, n in [(1, "#2a78d6", "Fist"), (2, "#eb6834", "Peace")]:
    mm = m & (labels == k)
    ax[0].scatter(neu_on[mm], glove_on[mm], color=col, s=22, label=n)
lim = [0.2, 0.9]
ax[0].plot(lim, lim, "k--", lw=.7)
ax[0].set(xlabel="Onset from brain (s after cue)", ylabel="Onset from glove (s after cue)", xlim=lim, ylim=lim,
          title=f"A. Brain-detected vs glove onset\nrho={r.statistic:.2f}; brain leads by {lead*1000:.0f} ms")
ax[0].legend(frameon=False)
ax[1].bar([0, 1], [a_cue.mean() * 100, a_on.mean() * 100], color=["#8a8a85", "#2a78d6"], width=.55)
for i, v in enumerate([a_cue.mean(), a_on.mean()]):
    ax[1].text(i, v * 100 + .4, f"{v*100:.1f}%", ha="center")
ax[1].set_xticks([0, 1], ["cue-locked\n(original windows)", "onset-locked\n(brain-detected)"])
ax[1].set(ylim=(85, 100), ylabel="3-class accuracy (%)", title="B. Same decoder, different time anchor (causal)")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D3_onset_locked.png", dpi=160)
