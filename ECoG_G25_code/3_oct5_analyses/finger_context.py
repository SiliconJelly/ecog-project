"""
Test 2 (Oct 5): finger patterns as CONTEXT.

Part 1  Is the pre-cue "I just moved" state neural memory, or leftover hand posture?
        - What does the hand look like in the last 1 s before the cue, by previous gesture?
        - Is the hand still moving then?
        - Decode "previous trial moved" from: brain only, glove only, both, and the brain
          AFTER removing everything the glove can explain (fit inside each training fold).
        - Regression: pre-cue motor beta ~ previous moved + finger posture + trial number.
Part 2  Gestures as finger combinations (the whole, not the pieces).
        - Finger vector of each gesture during the hold.
        - How well can the brain predict each finger's bend? (cross-validated ridge)
        - Inside fist/peace only: does the decoder's fist-vs-peace score track how much
          the index + middle fingers (the only fingers that differ) actually bent?
        - Do the fist <-> peace errors happen on trials with unusual index/middle posture?

All neural features use CAUSAL filtering (common.py).
"""
import json
import numpy as np
from scipy import stats
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
import common as C

fs = C.fs
rng = np.random.default_rng(1)
y, cue, glove, on, off, labels = C.load_raw()
prev = np.r_[0, labels[:-1]]
hp = np.arange(90) > 0
moved = (prev != 3).astype(int)
trialn = np.arange(90)
FC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GC = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
GN = {1: "Fist", 2: "Peace", 3: "Open"}
out = {}

# ---------------- glove summaries ----------------
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)   # glove smoothing only (behaviour, not neural)
pre_post = np.array([glove[:, o - fs:o].mean(1) for o in on])                # posture, last 1 s before cue
pre_vel = np.array([np.sqrt((np.diff(gs[:, o - fs:o], axis=1) ** 2).sum(0)).mean() * fs for o in on])  # finger speed
hold = np.array([glove[:, o + int(.75 * fs):o + int(1.75 * fs)].mean(1) for o in on])

print("Part 1a: pre-cue hand posture (0 straight - 1 bent), by previous gesture")
print("            " + "".join(f"{f:>8s}" for f in C.FINGERS) + "   speed")
tab = {}
for k in (1, 2, 3):
    m = prev == k
    tab[GN[k]] = pre_post[m].mean(0).tolist()
    print(f"  after {GN[k]:5s}" + "".join(f"{v:8.3f}" for v in pre_post[m].mean(0)) + f"   {pre_vel[m].mean():.3f}")
out["pre_posture_by_prev"] = tab
ps = {f: stats.kruskal(*[pre_post[prev == k, i] for k in (1, 2, 3)]).pvalue for i, f in enumerate(C.FINGERS)}
print("  Kruskal-Wallis p per finger:", {k: round(v, 4) for k, v in ps.items()})
out["pre_posture_kruskal_p"] = ps
u = stats.mannwhitneyu(pre_vel[hp & (moved == 1)], pre_vel[prev == 3])
print(f"  Pre-cue finger speed after movement vs after open: {pre_vel[hp & (moved==1)].mean():.3f} vs {pre_vel[prev==3].mean():.3f}  p={u.pvalue:.3f}")
out["pre_speed"] = dict(after_move=float(pre_vel[hp & (moved == 1)].mean()), after_open=float(pre_vel[prev == 3].mean()), p=float(u.pvalue))

# ---------------- neural pre-cue features (causal) ----------------
P = {b: C.band_power(b, True) for b in C.BANDS}
N = np.array([np.concatenate([C.win_log(P[b], o, -1, 0) for b in C.BANDS]) for o in on])   # 300 features
beta = np.array([10 * C.win_log(P["beta"], o, -1, 0, C.MOTOR).mean() for o in on])

def lda():
    return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def bal_cv(F, yy, seeds=range(5), residualize_on=None):
    """Balanced accuracy, 5-fold CV. If residualize_on is given, regress F on it
    INSIDE each training fold and decode the residual (test fold uses the training fit)."""
    accs = []
    for s in seeds:
        pred = np.zeros(len(yy), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(F, yy):
            Ftr, Fte = F[tr], F[te]
            if residualize_on is not None:
                Z = residualize_on
                reg = LinearRegression().fit(Z[tr], Ftr)
                Ftr, Fte = Ftr - reg.predict(Z[tr]), Fte - reg.predict(Z[te])
            pred[te] = lda().fit(Ftr, yy[tr]).predict(Fte)
        accs.append(balanced_accuracy_score(yy, pred))
    return float(np.mean(accs))

def perm_p(fun, yy, real, n=200):
    null = np.array([fun(rng.permutation(yy)) for _ in range(n)])
    return float((np.sum(null >= real) + 1) / (n + 1)), float(np.percentile(null, 95))

Z = np.c_[pre_post, trialn / 90]                      # posture (5 fingers) + session time
yy = moved[hp]
dec = {}
for name, F, res in [("Brain only", N[hp], None), ("Glove only", pre_post[hp], None),
                     ("Brain + glove", np.c_[N, pre_post][hp], None),
                     ("Brain, glove removed", N[hp], Z[hp])]:
    acc = bal_cv(F, yy, residualize_on=res)
    p, n95 = perm_p(lambda lab: bal_cv(F, lab, seeds=[0], residualize_on=res), yy, acc, n=200)
    dec[name] = dict(acc=acc, p=p, null95=n95)
    print(f"  Decode previous moved, {name:22s}: {acc*100:5.1f}% balanced  (shuffled 95th {n95*100:.0f}%, p={p:.3f})")
out["decode_prev_moved"] = dec

# Same question for previous fist vs peace: posture vs brain
fp = (prev == 1) | (prev == 2)
dec2 = {}
for name, F, res in [("Brain only", N[fp], None), ("Glove only", pre_post[fp], None),
                     ("Brain, glove removed", N[fp], Z[fp])]:
    acc = bal_cv(F, prev[fp], residualize_on=res)
    dec2[name] = acc
    print(f"  Decode previous fist vs peace, {name:22s}: {acc*100:5.1f}%")
out["decode_prev_fist_peace"] = dec2

# Regression on the single motor-beta number
def ols(Xm, v):
    Xm = np.c_[np.ones(len(v)), Xm]
    coef = np.linalg.lstsq(Xm, v, rcond=None)[0]
    r = v - Xm @ coef
    dfree = len(v) - Xm.shape[1]
    se = np.sqrt(np.diag(r @ r / dfree * np.linalg.inv(Xm.T @ Xm)))
    return coef, 2 * stats.t.sf(abs(coef / se), dfree), 1 - r.var() / v.var()

c0, p0, r0 = ols(np.c_[moved[hp], trialn[hp] / 90], beta[hp])
c1, p1, r1 = ols(np.c_[moved[hp], trialn[hp] / 90, pre_post[hp]], beta[hp])
c2, p2, r2 = ols(np.c_[trialn[hp] / 90, pre_post[hp]], beta[hp])
print(f"\n  Motor beta ~ previous moved + trial:            moved {c0[1]:+.2f} dB p={p0[1]:.4f}  R2={r0:.2f}")
print(f"  Motor beta ~ previous moved + trial + posture:  moved {c1[1]:+.2f} dB p={p1[1]:.4f}  R2={r1:.2f}")
print(f"  Motor beta ~ trial + posture only:                                     R2={r2:.2f}")
print("   finger coefs (dB per unit bend):", dict(zip(C.FINGERS, np.round(c1[3:], 2))), " p:", dict(zip(C.FINGERS, np.round(p1[3:], 3))))
out["beta_regression"] = dict(moved_only=dict(coef=c0[1], p=p0[1], r2=r0), with_posture=dict(coef=c1[1], p=p1[1], r2=r1),
                              posture_only_r2=r2, finger_coef=dict(zip(C.FINGERS, c1[3:].tolist())), finger_p=dict(zip(C.FINGERS, p1[3:].tolist())))
# within after-open trials only: does posture relate to beta at all?
op = prev == 3
rr = [stats.spearmanr(pre_post[op, i], beta[op]) for i in range(5)]
print("  Within after-OPEN trials, Spearman(finger posture, beta):", {f: (round(r.statistic, 2), round(r.pvalue, 3)) for f, r in zip(C.FINGERS, rr)})

# ---------------- Part 2: finger combinations ----------------
hg = C.band_power("highgamma", True)
X = C.decoder_features(hg, on)
print("\nPart 2a: hold posture per gesture (0 straight - 1 bent)")
fv = {}
for k in (1, 2, 3):
    fv[GN[k]] = hold[labels == k].mean(0).tolist()
    print(f"  {GN[k]:6s}" + "".join(f"{v:8.2f}" for v in hold[labels == k].mean(0)))
out["hold_finger_vectors"] = fv

print("\nPart 2b: predict each finger's bend during the hold from brain activity (cross-validated r)")
fr = {}
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))
for scope, mask in [("all trials", np.ones(90, bool)), ("fist/peace only", labels != 3)]:
    fr[scope] = {}
    for i, f in enumerate(C.FINGERS):
        rs = []
        for s in range(5):
            pr = cross_val_predict(ridge, X[mask], hold[mask, i], cv=KFold(10, shuffle=True, random_state=s))
            rs.append(stats.pearsonr(pr, hold[mask, i]).statistic)
        fr[scope][f] = float(np.mean(rs))
    print(f"  {scope:16s}" + "".join(f"  {f} {r:+.2f}" for f, r in fr[scope].items()))
out["finger_prediction_r"] = fr

# 2c: decoder fist-vs-peace score vs index+middle bend, within class
fpm = labels != 3
im_bend = hold[:, 1:3].mean(1)
score = np.zeros(90)
for s in range(20):
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(X[fpm], labels[fpm]):
        idx = np.where(fpm)[0]
        m = LDA(solver="lsqr", shrinkage="auto").fit(X[idx[tr]], labels[idx[tr]])
        score[idx[te]] += m.decision_function(X[idx[te]]) / 20     # >0 = peace
wc = {}
for k in (1, 2):
    m = labels == k
    r = stats.spearmanr(im_bend[m], score[m])
    wc[GN[k]] = dict(rho=r.statistic, p=r.pvalue)
    print(f"  Within {GN[k]}: Spearman(index+middle bend, decoder peace-score) rho={r.statistic:+.2f} p={r.pvalue:.3f}")
out["within_class_score_vs_bend"] = wc
# trials the fist/peace decoder gets wrong (score on wrong side)
wrong = fpm & (((labels == 1) & (score > 0)) | ((labels == 2) & (score < 0)))
print("  Fist/peace trials on the wrong side of the decoder:", (np.where(wrong)[0] + 1).tolist())
for i in np.where(wrong)[0]:
    k = labels[i]
    z = (im_bend[i] - im_bend[labels == k].mean()) / im_bend[labels == k].std()
    print(f"    trial {i+1} ({GN[k]}): index+middle bend {im_bend[i]:.2f} (class mean {im_bend[labels==k].mean():.2f}, z={z:+.1f})")
out["fp_wrong_trials"] = [int(i + 1) for i in np.where(wrong)[0]]

json.dump(out, open("results/finger_context.json", "w"), indent=2, default=float)

# ---------------- Figures ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
xs = np.arange(5)
for j, k in enumerate((1, 2, 3)):
    m = prev == k
    mu, se = pre_post[m].mean(0), pre_post[m].std(0) / np.sqrt(m.sum())
    ax[0].bar(xs + (j - 1) * .27, mu, .25, yerr=se, color=GC[k], label=f"after {GN[k]}")
ax[0].set_xticks(xs, C.FINGERS)
ax[0].set(ylabel="Finger bend, last 1 s before cue", title="A. The hand still 'remembers' the last gesture\n(pre-cue posture by previous gesture)")
ax[0].legend(frameon=False)
names = list(dec)
vals = [dec[n]["acc"] * 100 for n in names]
ax[1].bar(range(4), vals, color=["#2a78d6", "#eb6834", "#7a5cc4", "#1baf7a"])
for i, n in enumerate(names):
    ax[1].plot([i - .4, i + .4], [dec[n]["null95"] * 100] * 2, "k--", lw=1.2)
    ax[1].text(i, vals[i] + 1.5, f"{vals[i]:.0f}%\np={dec[n]['p']:.3f}", ha="center", fontsize=8)
ax[1].axhline(50, color="k", lw=.6)
ax[1].set_xticks(range(4), [n.replace(", ", ",\n") for n in names])
ax[1].set(ylim=(30, 105), ylabel="Balanced accuracy (%)", title="B. 'Was the previous trial a movement?'\nfrom the pre-cue window (dashed = shuffle 95th)")
lab = ["previous moved\n(no posture)", "previous moved\n(+ posture)"]
ax[2].bar([0, 1], [c0[1], c1[1]], color=["#8a8a85", "#2a78d6"], width=.55)
for i, p in enumerate([p0[1], p1[1]]):
    ax[2].text(i, [c0[1], c1[1]][i] + .05, f"p={p:.3f}", ha="center")
ax[2].set_xticks([0, 1], lab)
ax[2].axhline(0, color="k", lw=.6)
ax[2].set(ylabel="Pre-cue motor beta difference (dB)", title="C. Does the beta 'memory' survive\ncontrolling for finger posture?")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/T2a_posture_vs_neural.png", dpi=160)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
for k in (1, 2, 3):
    ax[0].plot(xs, hold[labels == k].mean(0), "-o", color=GC[k], lw=2, label=GN[k])
ax[0].axvspan(0.6, 2.4, color="grey", alpha=.12)
ax[0].text(1.5, 0.5, "index + middle:\nwhere fist and\npeace differ most", ha="center", fontsize=9)
ax[0].set_xticks(xs, C.FINGERS)
ax[0].set(ylabel="Finger bend during hold", ylim=(-.02, 1.05), title="A. Gestures as finger combinations")
ax[0].legend(frameon=False)
w = .38
for j, (scope, col) in enumerate(zip(fr, ["#8a8a85", "#2a78d6"])):
    ax[1].bar(xs + (j - .5) * w, list(fr[scope].values()), w - .03, color=col, label=scope)
ax[1].axhline(0, color="k", lw=.6)
ax[1].set_xticks(xs, C.FINGERS)
ax[1].set(ylabel="Cross-validated r (brain → finger bend)", ylim=(-.3, 1), title="B. Which fingers can the brain signal predict?")
ax[1].legend(frameon=False)
for k in (1, 2):
    m = labels == k
    ax[2].scatter(im_bend[m], score[m], color=GC[k], s=28, alpha=.8, label=GN[k])
ax[2].scatter(im_bend[wrong], score[wrong], s=120, facecolors="none", edgecolors="k", label="decoder wrong")
ax[2].axhline(0, color="k", lw=.6)
ax[2].set(xlabel="Index + middle bend during hold", ylabel="Decoder score (>0 = peace)",
          title="C. Decoder score vs the fingers that differ\n" + ", ".join(f"{g}: rho={wc[g]['rho']:+.2f} p={wc[g]['p']:.2f}" for g in wc))
ax[2].legend(frameon=False, fontsize=8)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/T2b_finger_combinations.png", dpi=160)
