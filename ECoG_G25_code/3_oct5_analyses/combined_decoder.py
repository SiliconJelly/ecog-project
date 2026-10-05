"""
Decoder test E (Oct 5): combine what worked.

  original       one LDA, 0.25-2.25 s (480 features)
  late fusion    LDA(closing 0.25-2.25 s) + LDA(opening 2.25-3.25 s), probabilities averaged
  fusion+shared  late fusion + a third LDA on the few channels whose fist-vs-peace effect is
                 the same in closing and opening (top 5, chosen INSIDE each training fold),
                 using their mean power in both windows (10 features)
Evaluated with 20 x 10-fold CV (same splits for all), and prospectively
(retrain every 10 trials on all earlier trials, predict trials 31-90).
Causal and zero-phase.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import common as C

y, cue, glove, on, off, labels = C.load_raw()
E_MAKE, E_REL = np.arange(0.25, 2.26, 0.25), np.arange(2.25, 3.26, 0.25)


def lda():
    return LDA(solver="lsqr", shrinkage="auto")


def shared_channels(Mw, Rw, yy, k=5):
    f, p = yy == 1, yy == 2
    t1 = stats.ttest_ind(Mw[f], Mw[p]).statistic
    t2 = stats.ttest_ind(Rw[f], Rw[p]).statistic
    score = np.where(np.sign(t1) == np.sign(t2), np.minimum(abs(t1), abs(t2)), 0)
    return np.argsort(score)[::-1][:k]


def predict(kind, tr, te, Xm, Xr, Mw, Rw):
    if kind == "original":
        return lda().fit(Xm[tr], labels[tr]).predict(Xm[te])
    pr = lda().fit(Xm[tr], labels[tr]).predict_proba(Xm[te]) + lda().fit(Xr[tr], labels[tr]).predict_proba(Xr[te])
    if kind == "fusion+shared":
        ch = shared_channels(Mw[tr], Rw[tr], labels[tr])
        S = np.c_[Mw[:, ch], Rw[:, ch]]
        pr = pr + lda().fit(S[tr], labels[tr]).predict_proba(S[te])
    return np.array([1, 2, 3])[pr.argmax(1)]


KINDS = ["original", "late fusion", "fusion+shared"]
out = {}
for causal in (True, False):
    tag = "causal" if causal else "zero-phase"
    hg = C.band_power(None, causal, lo_hi=(50, 300))
    Xm, Xr = C.decoder_features(hg, on, E_MAKE), C.decoder_features(hg, on, E_REL)
    Mw = np.array([C.win_log(hg, o, 0.5, 1.2) for o in on])     # closing window, per channel
    Rw = np.array([C.win_log(hg, o, 2.5, 3.2) for o in on])     # opening window, per channel
    res = {}
    for kind in KINDS:
        accs, CM = [], np.zeros((3, 3))
        for s in range(20):
            pred = np.zeros(90, int)
            for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(Xm, labels):
                pred[te] = predict(kind, tr, te, Xm, Xr, Mw, Rw)
            accs.append(np.mean(pred == labels)); CM += confusion_matrix(labels, pred)
        CM /= 20
        # prospective
        c = []
        for s0 in range(30, 90, 10):
            tr, te = np.arange(s0), np.arange(s0, min(s0 + 10, 90))
            c.extend(predict(kind, tr, te, Xm, Xr, Mw, Rw) == labels[te])
        res[kind] = dict(acc=float(np.mean(accs)), per_run=accs, errors=float(CM.sum() - np.trace(CM)),
                         fist_peace=float(CM[0, 1] + CM[1, 0]), peace_open=float(CM[1, 2] + CM[2, 1]),
                         confusion=CM.tolist(), prospective=float(np.mean(c)))
        print(f"[{tag}] {kind:14s} CV {np.mean(accs)*100:5.1f}%  errors/run {res[kind]['errors']:.2f} "
              f"(fist<->peace {res[kind]['fist_peace']:.2f}, peace<->open {res[kind]['peace_open']:.2f})  "
              f"prospective {np.mean(c)*100:.1f}%", flush=True)
    for kind in KINDS[1:]:
        d = np.array(res[kind]["per_run"]) - np.array(res["original"]["per_run"])
        res[kind]["runs_better_vs_original"] = int(np.sum(d > 0)); res[kind]["runs_worse_vs_original"] = int(np.sum(d < 0))
        print(f"  {kind}: better than original in {np.sum(d>0)}/20 runs, worse in {np.sum(d<0)}/20")
    out[tag] = res
json.dump(out, open("results/combined_decoder.json", "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
cols = ["#8a8a85", "#1baf7a", "#2a78d6"]
r = out["causal"]
for i, k in enumerate(KINDS):
    ax[0].bar(i - .2, r[k]["acc"] * 100, .38, color=cols[i])
    ax[0].bar(i + .2, r[k]["prospective"] * 100, .38, color=cols[i], alpha=.45, hatch="//", edgecolor="w")
    ax[0].text(i - .2, r[k]["acc"] * 100 + .5, f"{r[k]['acc']*100:.1f}", ha="center", fontsize=8)
    ax[0].text(i + .2, r[k]["prospective"] * 100 + .5, f"{r[k]['prospective']*100:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(range(3), KINDS)
ax[0].set(ylim=(75, 100), ylabel="3-class accuracy (%)", title="A. Causal: random CV (solid) vs predicting\nonly future trials (hatched)")
for i, k in enumerate(KINDS):
    ax[1].bar(i, r[k]["fist_peace"], color=cols[i])
    ax[1].text(i, r[k]["fist_peace"] + .05, f"{r[k]['fist_peace']:.2f}", ha="center")
ax[1].set_xticks(range(3), KINDS)
ax[1].set(ylabel="Fist↔peace errors per run (of 90)", title="B. The hard pair")
CMb = np.array(r[max(KINDS, key=lambda k: r[k]["acc"])]["confusion"])
ax[2].imshow(CMb, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax[2].text(j, i, f"{CMb[i,j]:.1f}", ha="center", va="center", color="white" if CMb[i, j] > 15 else "k")
ax[2].set_xticks(range(3), ["Fist", "Peace", "Open"]); ax[2].set_yticks(range(3), ["Fist", "Peace", "Open"])
best = max(KINDS, key=lambda k: r[k]["acc"])
ax[2].set(xlabel="Predicted", ylabel="True", title=f"C. Best (causal): {best}, {r[best]['acc']*100:.1f}%")
for a in ax[:2]:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D6_combined_decoder.png", dpi=160)
