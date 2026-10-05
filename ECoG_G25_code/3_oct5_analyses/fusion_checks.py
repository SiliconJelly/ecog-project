"""
Checks requested Oct 5 (causal preprocessing throughout):

1. PAIRED comparison, original vs late fusion, on identical held-out trials.
   - Random CV: 20 x 10-fold, same folds; per-trial correctness saved for both.
     Paired sign-flip permutation test on the per-trial accuracy difference.
   - Chronological: retrain every 10 trials on all earlier trials, predict the next 10.
     Exact McNemar test on the discordant trials.
2. ACCURACY vs DELAY: when must the decision be made?
   Closing-only decoder using 0.25 s -> T (T = 0.75 ... 2.25 s), and fusion with an
   opening model using 2.25 s -> T (T = 2.5 ... 3.25 s). Hand starts opening ~2.47 s.
3. RETRAINING: adaptation or just more data? Every test block (trials 31-90, blocks of 10)
   is predicted by decoders trained on the SAME NUMBER of trials (30):
   first 30 (early) | last 30 before the block (recent) | 30 random earlier trials.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import common as C

rng = np.random.default_rng(17)
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(50, 300))
lda = lambda: LDA(solver="lsqr", shrinkage="auto")


def feats(t0, t1):
    edges = np.arange(t0, t1 + 1e-9, 0.25)
    return C.decoder_features(hg, on, edges)


Xm, Xr = feats(0.25, 2.25), feats(2.25, 3.25)


def pred_orig(tr, te, Xa=Xm):
    return lda().fit(Xa[tr], labels[tr]).predict(Xa[te])


def pred_fus(tr, te, Xa=Xm, Xb=Xr):
    p = lda().fit(Xa[tr], labels[tr]).predict_proba(Xa[te]) + lda().fit(Xb[tr], labels[tr]).predict_proba(Xb[te])
    return np.array([1, 2, 3])[p.argmax(1)]


out = {}
# ---- 1a paired random CV ----
co, cf = np.zeros(90), np.zeros(90)
P_o, P_f = [], []
for s in range(20):
    po, pf = np.zeros(90, int), np.zeros(90, int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(Xm, labels):
        po[te], pf[te] = pred_orig(tr, te), pred_fus(tr, te)
    co += (po == labels) / 20; cf += (pf == labels) / 20
    P_o.append(po.tolist()); P_f.append(pf.tolist())
d = cf - co
obs = d.mean()
null = np.array([(d * rng.choice([-1, 1], 90)).mean() for _ in range(20000)])
p_pair = float((np.sum(null >= obs) + 1) / 20001)
print(f"1a. Random CV, identical folds: original {co.mean()*100:.1f}%  fusion {cf.mean()*100:.1f}%  "
      f"diff {obs*100:+.2f} pts; trials helped {np.sum(d>0)}, hurt {np.sum(d<0)}; paired sign-flip p={p_pair:.4f}")
helped, hurt = np.where(d > 0.01)[0] + 1, np.where(d < -0.01)[0] + 1
print(f"    trials helped: {helped.tolist()}  hurt: {hurt.tolist()}")
out["random_cv"] = dict(original=float(co.mean()), fusion=float(cf.mean()), diff=float(obs), p_signflip=p_pair,
                        trials_helped=helped.tolist(), trials_hurt=hurt.tolist(),
                        per_trial_correct_original=co.tolist(), per_trial_correct_fusion=cf.tolist())

# ---- 1b chronological paired ----
ro, rf = np.full(90, np.nan), np.full(90, np.nan)
for s0 in range(30, 90, 10):
    tr, te = np.arange(s0), np.arange(s0, min(s0 + 10, 90))
    ro[te], rf[te] = pred_orig(tr, te) == labels[te], pred_fus(tr, te) == labels[te]
m = slice(30, 90)
b = int(np.sum((ro[m] == 0) & (rf[m] == 1)))   # fusion fixes
c = int(np.sum((ro[m] == 1) & (rf[m] == 0)))   # fusion breaks
p_mc = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
print(f"1b. Chronological (trials 31-90): original {np.nanmean(ro)*100:.1f}%  fusion {np.nanmean(rf)*100:.1f}%; "
      f"fusion fixes {b}, breaks {c}; exact McNemar p={p_mc:.3f}")
out["chronological"] = dict(original=float(np.nanmean(ro)), fusion=float(np.nanmean(rf)), fixes=b, breaks=c, p_mcnemar=p_mc)


# ---- 2 accuracy vs delay ----
def cv_acc(fun, runs=10):
    a = []
    for s in range(runs):
        pr = np.zeros(90, int)
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(Xm, labels):
            pr[te] = fun(tr, te)
        a.append(np.mean(pr == labels))
    return float(np.mean(a))


delay = []
for T in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]:
    Xa = feats(0.25, T)
    acc = cv_acc(lambda tr, te: pred_orig(tr, te, Xa))
    delay.append(dict(decoder="closing only", decision_s=T, acc=acc))
    print(f"2. closing only, decide at {T:.2f} s: {acc*100:.1f}%", flush=True)
for T in [2.5, 2.75, 3.0, 3.25]:
    Xb = feats(2.25, T)
    acc = cv_acc(lambda tr, te: pred_fus(tr, te, Xm, Xb))
    delay.append(dict(decoder="closing + opening", decision_s=T, acc=acc))
    print(f"2. fusion, decide at {T:.2f} s: {acc*100:.1f}%", flush=True)
out["accuracy_vs_delay"] = delay

# ---- 3 retraining: same size (30), early vs recent vs random earlier ----
res3 = {"early (trials 1-30)": [], "recent (last 30)": [], "random 30 earlier": []}
for s0 in range(30, 90, 10):
    te = np.arange(s0, min(s0 + 10, 90))
    for name, tr in [("early (trials 1-30)", np.arange(30)), ("recent (last 30)", np.arange(s0 - 30, s0))]:
        res3[name].extend(pred_orig(tr, te) == labels[te])
    rr = []
    for _ in range(100):
        tr = rng.choice(np.arange(s0), 30, replace=False)
        rr.append(np.mean(pred_orig(tr, te) == labels[te]))
    res3["random 30 earlier"].append(float(np.mean(rr)))
early, recent = np.array(res3["early (trials 1-30)"], float), np.array(res3["recent (last 30)"], float)
rand = float(np.mean(res3["random 30 earlier"]))
b3 = int(np.sum((early == 0) & (recent == 1))); c3 = int(np.sum((early == 1) & (recent == 0)))
p3 = float(stats.binomtest(b3, b3 + c3, .5).pvalue) if b3 + c3 else 1.0
print(f"3. Same training size (30), predicting trials 31-90: early {early.mean()*100:.1f}%  "
      f"recent {recent.mean()*100:.1f}%  random-earlier {rand*100:.1f}%; recent vs early McNemar p={p3:.3f} ({b3} fixed, {c3} broken)")
out["retraining_same_size"] = dict(early=float(early.mean()), recent=float(recent.mean()), random_earlier=rand,
                                   recent_vs_early_fixed=b3, recent_vs_early_broken=c3, p_mcnemar=p3)
json.dump(dict(out, paired_predictions=dict(original=P_o, fusion=P_f)), open("results/fusion_checks.json", "w"), indent=2)

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
cl = [d_ for d_ in delay if d_["decoder"] == "closing only"]
fu = [d_ for d_ in delay if d_["decoder"] != "closing only"]
ax[0].plot([d_["decision_s"] for d_ in cl], [d_["acc"] * 100 for d_ in cl], "-o", color="#8a8a85", lw=2, label="closing only")
ax[0].plot([cl[-1]["decision_s"]] + [d_["decision_s"] for d_ in fu], [cl[-1]["acc"] * 100] + [d_["acc"] * 100 for d_ in fu],
           "-o", color="#1baf7a", lw=2, label="+ opening model (fusion)")
ax[0].axvline(2.0, color="k", ls="--", lw=.8); ax[0].text(2.02, 80, "cue off", fontsize=8)
ax[0].axvline(2.47, color="#eb6834", ls=":", lw=1.2); ax[0].text(2.5, 80, "hand starts\nopening", fontsize=8, color="#eb6834")
ax[0].set(xlabel="Decision time (s after cue)", ylabel="3-class accuracy (%)", ylim=(75, 100),
          title="A. Accuracy vs how long the decoder waits")
ax[0].legend(frameon=False, loc="lower right")
xs = np.arange(90) + 1
ax[1].scatter(xs[d > 0], d[d > 0] * 100, color="#1baf7a", s=25, label="fusion more often right")
ax[1].scatter(xs[d < 0], d[d < 0] * 100, color="#eb6834", s=25, label="original more often right")
ax[1].axhline(0, color="k", lw=.6)
ax[1].set(xlabel="Trial", ylabel="Δ % of CV runs correct", title=f"B. Same held-out trials, paired\n{np.sum(d>0)} trials helped, {np.sum(d<0)} hurt; p={p_pair:.3f}")
ax[1].legend(frameon=False, fontsize=8)
names = ["early (trials 1-30)", "random 30 earlier", "recent (last 30)"]
vals = [early.mean(), rand, recent.mean()]
ax[2].bar(range(3), [v * 100 for v in vals], color=["#eb6834", "#c3c2b7", "#2a78d6"])
for i, v in enumerate(vals):
    ax[2].text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center")
ax[2].set_xticks(range(3), ["first 30\n(early)", "30 random\nearlier", "last 30\n(recent)"])
ax[2].set(ylim=(50, 105), ylabel="Accuracy on trials 31-90 (%)", title=f"C. Same training size (30): does recency help?\nrecent vs early p={p3:.3f}")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D7_fusion_checks.png", dpi=160)
