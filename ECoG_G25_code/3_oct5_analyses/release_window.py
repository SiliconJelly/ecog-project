"""
Decoder test A (Oct 5): add the RELEASE window.

The original decoder reads high-gamma 0.25-2.25 s after the cue. The hand starts
opening ~2.47 s after the cue, and Tests 3/4 showed the opening phase carries
fist-vs-peace information in a DIFFERENT pattern. So: does adding 2.25-3.25 s help?

Variants (same 20 x 10-fold CV splits for all):
  original      0.25-2.25 s, 8 bins (480 features)       <- the 93.3% decoder
  + release     0.25-3.25 s, 12 bins (720 features), one LDA
  release only  2.25-3.25 s, 4 bins
  late fusion   one LDA on 0.25-2.25 s + one LDA on 2.25-3.25 s, average their probabilities
Control: '+ release' with the release features shuffled across trials (100x), so
any gain must beat simply adding 240 extra features of noise.
Run for zero-phase (original) and causal preprocessing.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import common as C

N_SHUF = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RUNS = 20
rng = np.random.default_rng(3)
y, cue, glove, on, off, labels = C.load_raw()
E_MAKE = np.arange(0.25, 2.26, 0.25)
E_REL = np.arange(2.25, 3.26, 0.25)


def lda():
    return LDA(solver="lsqr", shrinkage="auto")


def run(Fs, runs=RUNS, fusion=False, mask=None):
    """Fs: one feature matrix, or a list of matrices for late fusion. Returns mean accuracy,
    mean confusion matrix, per-run accuracies."""
    m = np.ones(90, bool) if mask is None else mask
    yy = labels[m]
    k = len(np.unique(yy))
    accs, CM = [], np.zeros((k, k))
    for s in range(runs):
        pred = np.zeros(len(yy), int)
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(yy, yy):
            if fusion:
                pr = sum(lda().fit(F[m][tr], yy[tr]).predict_proba(F[m][te]) for F in Fs)
                pred[te] = np.unique(yy)[pr.argmax(1)]
            else:
                pred[te] = lda().fit(Fs[m][tr], yy[tr]).predict(Fs[m][te])
        accs.append(np.mean(pred == yy))
        CM += confusion_matrix(yy, pred, labels=np.unique(yy))
    return float(np.mean(accs)), CM / runs, np.array(accs)


def fp_err(CM):
    return CM[0, 1] + CM[1, 0]


results = {}
for causal in (False, True):
    tag = "causal" if causal else "zero-phase"
    hg = C.band_power(None, causal, lo_hi=(50, 300))
    Xm = C.decoder_features(hg, on, E_MAKE)
    Xr = C.decoder_features(hg, on, E_REL)
    Xb = np.c_[Xm, Xr]
    R = {}
    for name, F, fus in [("original", Xm, False), ("+ release", Xb, False),
                         ("release only", Xr, False), ("late fusion", [Xm, Xr], True)]:
        acc, CM, per = run(F, fusion=fus)
        accfp, CMfp, perfp = run(F, fusion=fus, mask=labels != 3)
        R[name] = dict(acc=acc, fist_peace_errors=fp_err(CM), total_errors=float(CM.sum() - np.trace(CM)),
                       confusion=CM.tolist(), fist_vs_peace_acc=accfp, _per=per, _perfp=perfp)
        print(f"[{tag}] {name:13s} 3-class {acc*100:5.1f}%  errors/run {CM.sum()-np.trace(CM):4.2f} "
              f"(fist<->peace {fp_err(CM):4.2f})   fist-vs-peace {accfp*100:5.1f}%", flush=True)
    # shuffle control for the best combined variant: release features permuted across trials
    best = max(["+ release", "late fusion"], key=lambda n: R[n]["acc"])
    gain = R[best]["acc"] - R["original"]["acc"]
    null = []
    for _ in range(N_SHUF):
        Xr_s = Xr[rng.permutation(90)]
        F = [Xm, Xr_s] if best == "late fusion" else np.c_[Xm, Xr_s]
        null.append(run(F, runs=5, fusion=best == "late fusion")[0] - R["original"]["acc"])
    null = np.array(null)
    p = float((np.sum(null >= gain) + 1) / (N_SHUF + 1))
    # paired: per-run difference on the identical splits
    d = R[best]["_per"] - R["original"]["_per"]
    print(f"[{tag}] best combined = {best}: gain {gain*100:+.2f} pts; shuffled-release gain mean {null.mean()*100:+.2f}, "
          f"95th {np.percentile(null,95)*100:+.2f}; p={p:.3f}; runs improved {np.sum(d>0)}/{RUNS}, worse {np.sum(d<0)}", flush=True)
    R["_control"] = dict(best=best, gain=gain, null_mean=float(null.mean()), null95=float(np.percentile(null, 95)), p=p,
                         runs_better=int(np.sum(d > 0)), runs_worse=int(np.sum(d < 0)))
    results[tag] = R

json.dump({t: {k: ({a: b for a, b in v.items() if not a.startswith("_")} if isinstance(v, dict) else v)
               for k, v in R.items()} for t, R in results.items()}, open("results/release_window.json", "w"), indent=2, default=float)

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
names = ["original", "+ release", "late fusion", "release only"]
cols = ["#8a8a85", "#2a78d6", "#1baf7a", "#eb6834"]
for j, tag in enumerate(["zero-phase", "causal"]):
    R = results[tag]
    xs = np.arange(4) + (j - .5) * .38
    ax[0].bar(xs, [R[n]["acc"] * 100 for n in names], .35, color=cols, alpha=1 if j else .55, edgecolor="k", lw=.4)
    ax[1].bar(xs, [R[n]["fist_peace_errors"] for n in names], .35, color=cols, alpha=1 if j else .55, edgecolor="k", lw=.4)
ax[0].set_xticks(range(4), names)
ax[0].set(ylim=(60, 100), ylabel="3-class accuracy (%)", title="A. Accuracy (left of pair = zero-phase, right = causal)")
ax[1].set_xticks(range(4), names)
ax[1].set(ylabel="Fist↔peace errors per run (of 90 trials)", title="B. The hard pair")
R = results["causal"]
CMb = np.array(R[R["_control"]["best"]]["confusion"])
im = ax[2].imshow(CMb, cmap="Blues")
for i in range(3):
    for k in range(3):
        ax[2].text(k, i, f"{CMb[i,k]:.1f}", ha="center", va="center", color="white" if CMb[i, k] > 15 else "k")
ax[2].set_xticks(range(3), ["Fist", "Peace", "Open"]); ax[2].set_yticks(range(3), ["Fist", "Peace", "Open"])
ax[2].set(xlabel="Predicted", ylabel="True",
          title=f"C. Causal, {R['_control']['best']}: {R[R['_control']['best']]['acc']*100:.1f}%\n"
                f"gain vs original {R['_control']['gain']*100:+.1f} pts, shuffle p={R['_control']['p']:.3f}")
for a in ax[:2]:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D1_release_window.png", dpi=160)
