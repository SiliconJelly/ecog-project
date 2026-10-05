"""
Decoder test B (Oct 5): robustness over time.

Random 10-fold CV mixes early and late trials, so it can hide drift. Here the
decoder only ever predicts trials that come AFTER its training data, like a real BCI.

Schemes (original 480 high-gamma features, causal preprocessing; zero-phase in the JSON too):
  random CV        reference (93% decoder)
  fixed            train once on trials 1-30, never update, predict 31-90
  fixed + recentre fixed decoder, but each feature is re-centred using the mean of the
                   previous 15 trials (no labels needed: unsupervised drift correction)
  expanding        retrain every 10 trials on ALL earlier labelled trials (supervised recalibration)
  sliding          retrain every 10 trials on only the LAST 30 trials
  half split       first 45 -> last 45, and last 45 -> first 45
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import common as C

y, cue, glove, on, off, labels = C.load_raw()


def lda():
    return LDA(solver="lsqr", shrinkage="auto")


def recentre(X, start, window=15):
    """Subtract, from every trial t >= start, the mean of trials t-window..t-1 (labels unused),
    and from training trials the mean of the training set."""
    Xc = X.copy()
    for t in range(start, len(X)):
        Xc[t] = X[t] - X[max(0, t - window):t].mean(0)
    return Xc


def schemes(X):
    out, curves = {}, {}
    pred = np.mean([cross_val_predict(lda(), X, labels, cv=StratifiedKFold(10, shuffle=True, random_state=s)) == labels
                    for s in range(20)], axis=0)
    out["random CV"] = float(pred[30:].mean())          # same trials 31-90 for a fair comparison
    out["random CV (all 90)"] = float(pred.mean())
    curves["random CV"] = pred
    # fixed
    m = lda().fit(X[:30], labels[:30])
    c = (m.predict(X[30:]) == labels[30:]).astype(float)
    out["fixed"] = float(c.mean()); curves["fixed"] = np.r_[np.full(30, np.nan), c]
    # fixed + unsupervised recentring
    Xc = recentre(X, 30)
    Xtr = X[:30] - X[:30].mean(0)
    m = lda().fit(Xtr, labels[:30])
    c = (m.predict(Xc[30:]) == labels[30:]).astype(float)
    out["fixed + recentre"] = float(c.mean()); curves["fixed + recentre"] = np.r_[np.full(30, np.nan), c]
    # expanding / sliding supervised recalibration
    for name, win in [("expanding", None), ("sliding (last 30)", 30)]:
        c = np.full(90, np.nan)
        for s in range(30, 90, 10):
            tr = np.arange(0 if win is None else s - win, s)
            m = lda().fit(X[tr], labels[tr])
            te = np.arange(s, min(s + 10, 90))
            c[te] = m.predict(X[te]) == labels[te]
        out[name] = float(np.nanmean(c)); curves[name] = c
    # half splits
    m = lda().fit(X[:45], labels[:45]); a1 = float(np.mean(m.predict(X[45:]) == labels[45:]))
    m = lda().fit(X[45:], labels[45:]); a2 = float(np.mean(m.predict(X[:45]) == labels[:45]))
    out["first half -> second"], out["second half -> first"] = a1, a2
    return out, curves


res, curves_all = {}, {}
for causal in (True, False):
    tag = "causal" if causal else "zero-phase"
    hg = C.band_power(None, causal, lo_hi=(50, 300))
    X = C.decoder_features(hg, on)
    res[tag], curves_all[tag] = schemes(X)
    print(f"[{tag}]")
    for k, v in res[tag].items():
        n = 90 if "all 90" in k else (45 if "half" in k else 60)
        lo, hi = stats.binomtest(int(round(v * n)), n).proportion_ci(0.95)
        print(f"  {k:24s} {v*100:5.1f}%   (95% CI {lo*100:.0f}-{hi*100:.0f}%, n={n})")
    # drift: overall feature level over the session
    lvl = X.reshape(90, 8, 60).mean((1, 2))
    r = stats.spearmanr(np.arange(90), lvl)
    res[tag]["drift_spearman"] = dict(rho=float(r.statistic), p=float(r.pvalue))
    print(f"  mean log high-gamma vs trial number: rho={r.statistic:+.2f}, p={r.pvalue:.4f}")
    res[tag]["_lvl"] = lvl

json.dump({t: {k: v for k, v in r.items() if not k.startswith("_")} for t, r in res.items()},
          open("results/robust_time.json", "w"), indent=2)

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
r = res["causal"]
names = ["random CV", "fixed", "fixed + recentre", "sliding (last 30)", "expanding"]
cols = ["#c3c2b7", "#eb6834", "#eda100", "#1baf7a", "#2a78d6"]
ax[0].bar(range(5), [r[n] * 100 for n in names], color=cols)
for i, n in enumerate(names):
    ax[0].text(i, r[n] * 100 + 1, f"{r[n]*100:.0f}%", ha="center")
ax[0].set_xticks(range(5), ["random\nCV", "fixed\n(train 1-30)", "fixed +\nrecentre", "sliding\nlast 30", "expanding\nretrain"])
ax[0].set(ylim=(50, 105), ylabel="Accuracy on trials 31-90 (%)", title="A. Predicting only the future (causal)")
cv = curves_all["causal"]
k = 10
for n, col in zip(names, cols):
    c = cv[n]
    xs = np.arange(30, 90, k)
    ax[1].plot(xs + k / 2 + 1, [np.nanmean(c[s:s + k]) * 100 for s in xs], "-o", color=col, label=n, lw=2)
ax[1].set(xlabel="Trial (block of 10)", ylabel="Accuracy in block (%)", ylim=(40, 105), title="B. Accuracy across the session")
ax[1].legend(frameon=False, fontsize=8)
ctrl = json.load(open("results/robust_control.json"))["causal"]
vals = [ctrl["fixed_time_ordered"], ctrl["random30_mean"], r["expanding"], ctrl["expanding_size_matched_random_mean"]]
ax[2].bar([0, 1, 3, 4], [v * 100 for v in vals], color=["#eb6834", "#c3c2b7", "#2a78d6", "#c3c2b7"], width=.8)
lo, hi = ctrl["random30_5_95"]
ax[2].errorbar([1], [ctrl["random30_mean"] * 100], yerr=[[ctrl["random30_mean"] * 100 - lo * 100], [hi * 100 - ctrl["random30_mean"] * 100]], color="k", capsize=4)
for x, v in zip([0, 1, 3, 4], vals):
    ax[2].text(x, v * 100 + 1, f"{v*100:.0f}%", ha="center")
ax[2].set_xticks([0, 1, 3, 4], ["first 30\n(time order)", "30 random\ntrials", "expanding\n(time order)", "same sizes,\nrandom"])
ax[2].set(ylim=(50, 105), ylabel="Accuracy (%)", title=f"C. Is it time, or too few training trials?\nfixed vs random-30: p={ctrl['p_time_worse_than_random']:.2f}")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D2_robust_time.png", dpi=160)
