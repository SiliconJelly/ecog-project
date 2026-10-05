"""
Temporal generalization from REST through the TRANSITION into MOVEMENT (Oct 5).

Question: does the in-between (rest -> transition) already contain the pattern the
movement will use, i.e. does a decoder trained early generalize FORWARD in time?

  - Train an LDA at every 100 ms bin from -1.0 s to +2.5 s around the cue (current gesture),
    test it at every bin. Diagonal = 'when is the gesture decodable'; off-diagonal =
    'is the pattern at time A reused at time B'.
  - Causal filtering (nothing after a time point can leak into it).
  - Same CV folds at every train/test time; 5-fold x 3 repeats.
  - Null: shuffle the gesture labels, recompute the whole matrix with the SAME 3 repeats,
    100 times. Used for (a) the first time the diagonal beats chance, and (b) block tests.

Blocks (seconds after cue; hand starts moving ~0.48 s):
  REST        -1.0 .. 0.0     (cues are random, so this should NOT predict the next gesture)
  TRANSITION   0.1 .. 0.5     (cue seen, hand not yet moving)
  MOVEMENT     0.6 .. 1.6     (closing + hold)
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import common as C

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 100
fs, step = C.fs, 0.1
times = np.round(np.arange(-1.0, 2.5, step), 2)            # bin START times
rng = np.random.default_rng(51)
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(52, 300))
X_all = np.stack([np.log10(np.stack([hg[:, o + int(round(t * fs)): o + int(round((t + step) * fs))].mean(1)
                                     for t in times], axis=1)) for o in on])      # trials x 60 ch x bins
BLOCKS = {"REST": (-1.0, 0.0), "TRANSITION": (0.1, 0.5), "MOVEMENT": (0.6, 1.6)}
sel = {k: np.where((times >= a - 1e-9) & (times < b - 1e-9))[0] for k, (a, b) in BLOCKS.items()}


def tg(X, yy, repeats=3):
    n = len(times)
    M = np.zeros((n, n))
    for s in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X[:, :, 0], yy):
            for i in range(n):
                m = LDA(solver="lsqr", shrinkage="auto").fit(X[tr, :, i], yy[tr])
                P = np.stack([m.predict(X[te, :, j]) for j in range(n)], 1)
                M[i] += (P == yy[te, None]).sum(0)
    return M / (len(yy) * repeats)


def block(M, a, b):
    return M[np.ix_(sel[a], sel[b])].mean()


PAIRS = [("REST", "REST"), ("TRANSITION", "TRANSITION"), ("MOVEMENT", "MOVEMENT"),
         ("REST", "MOVEMENT"), ("TRANSITION", "MOVEMENT"), ("MOVEMENT", "TRANSITION")]
out, mats = {}, {}
for task, keep, chance in [("3-class", labels > 0, 1 / 3), ("fist-vs-peace", labels != 3, 0.5)]:
    X, yy = X_all[keep], labels[keep]
    M = tg(X, yy)
    null = np.array([tg(X, rng.permutation(yy)) for _ in range(N_PERM)])     # perms x bins x bins
    d95 = np.percentile(np.stack([np.diag(n) for n in null]), 95, axis=0)
    above = np.diag(M) > d95
    first = next((times[i] for i in range(len(times) - 1) if above[i] and above[i + 1] and times[i] >= -1), None)
    res = {"chance": chance, "first_decodable_bin_start_s": None if first is None else float(first),
           "peak_diag": float(np.diag(M).max()), "peak_time_s": float(times[np.argmax(np.diag(M))]), "blocks": {}}
    print(f"\n[{task}] chance {chance*100:.0f}%  ({N_PERM} label shuffles, 3 CV repeats for real and null)")
    print(f"  diagonal first beats shuffle 95th (2 bins in a row) at {first} s; peak {res['peak_diag']*100:.0f}% at {res['peak_time_s']:+.1f} s")
    for a, b in PAIRS:
        v = block(M, a, b); nv = np.array([block(n, a, b) for n in null])
        p = float((np.sum(nv >= v) + 1) / (N_PERM + 1))
        res["blocks"][f"{a}->{b}"] = dict(acc=float(v), null95=float(np.percentile(nv, 95)), p=p)
        print(f"  train {a:10s} -> test {b:10s} {v*100:5.1f}%   shuffle 95th {np.percentile(nv,95)*100:4.1f}%   p={p:.3f}")
    out[task] = res
    mats[task] = (M, d95, chance)
json.dump(out, open("results/tg_transition.json", "w"), indent=2)
np.savez("results/tg_transition_mats.npz", **{k.replace("-", "_"): v[0] for k, v in mats.items()})

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.1]})
ext = [times[0], times[-1] + step, times[0], times[-1] + step]
for a_, (task, (M, d95, chance)) in zip(ax[:2], mats.items()):
    im = a_.imshow(M, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=chance, vmin=0, vmax=1), extent=ext)
    for v, ls in [(0, "--"), (0.48, ":")]:
        a_.axvline(v, color="k", lw=.8, ls=ls); a_.axhline(v, color="k", lw=.8, ls=ls)
    a_.plot([times[0], times[-1]], [times[0], times[-1]], color="k", lw=.4)
    a_.set(xlabel="Test time (s from cue)", ylabel="Train time (s from cue)",
           title=f"{task}: train at one moment, test at every other\n(dashed = cue, dotted = hand starts moving)")
    fig.colorbar(im, ax=a_, shrink=.8, label="Accuracy")
for task, col in [("3-class", "#2a78d6"), ("fist-vs-peace", "#eb6834")]:
    M, d95, chance = mats[task]
    ax[2].plot(times + step / 2, np.diag(M) * 100, color=col, lw=2, label=f"{task} (decode at same time)")
    ax[2].plot(times + step / 2, d95 * 100, color=col, lw=1, ls="--", label=f"{task} shuffle 95th")
ax[2].axvline(0, color="k", lw=.8, ls="--"); ax[2].axvline(0.48, color="k", lw=.8, ls=":")
for name, (a, b) in BLOCKS.items():
    ax[2].axvspan(a, b, color={"REST": "#c3c2b7", "TRANSITION": "#eda100", "MOVEMENT": "#1baf7a"}[name], alpha=.15)
    ax[2].text((a + b) / 2, 102, name.lower(), ha="center", fontsize=8)
ax[2].set(xlabel="Time from cue (s)", ylabel="Accuracy (%)", ylim=(20, 106), title="When does the gesture appear?")
ax[2].legend(frameon=False, fontsize=8, loc="center right")
ax[2].spines[["top", "right"]].set_visible(False)
fig.savefig("figures/E2_tg_transition.png", dpi=160)
