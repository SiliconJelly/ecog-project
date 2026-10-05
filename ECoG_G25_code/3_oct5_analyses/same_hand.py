"""
Same hand, different meaning (Oct 5, partner's idea).

Is an open hand that MEANS "paper" different in the brain from an open hand that is
just resting, when the finger posture is the same?

Windows (0.75 s each, causal filtering, every pair comes from the SAME trial):
  REST        -0.75 .. 0    s   relaxed hand before the cue (no instruction yet)
  PAPER        1.00 .. 1.75 s   open-hand trials, holding "paper" while the cue is on screen
  OPEN-AFTER   3.25 .. 4.00 s   open-hand trials, cue gone, nothing was done with the hand
  LOOSE        3.25 .. 4.00 s   fist/peace trials, hand has re-opened after the gesture

Comparisons (paired, within trial):
  1. PAPER vs REST          instructed open vs natural open            (30 open trials)
  2. PAPER vs OPEN-AFTER    same trial, same hand, cue on vs cue off    (30 open trials)
  3. OPEN-AFTER vs REST     cue gone, no gesture made: should look like rest if the
                            difference in 1 is tied to the instruction  (30 open trials)
  4. LOOSE vs REST          just-released hand vs natural rest          (60 fist/peace trials)

Controls (all must hold before a brain difference counts):
  a. Posture matched: pairs whose glove posture differs by more than a caliper are dropped.
  b. Glove-only decoder on the kept pairs must be near chance: if the glove can tell the
     two windows apart, the hand was NOT the same and the comparison is confounded.
     Glove speed (movement) is included, not just posture.
  c. Cross-validation groups both windows of a trial together (GroupKFold), so no trial is
     ever split between training and testing.
  d. Null: randomly swap the two labels WITHIN each pair (the matched null for a paired
     design), re-running the full CV, 200 times, same number of CV repeats as the real score.
  e. Univariate check on two interpretable numbers: motor high-gamma and motor beta (dB),
     paired Wilcoxon.
"""
import json
import numpy as np
from scipy import stats
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
import common as C

fs = C.fs
rng = np.random.default_rng(31)
y, cue, glove, on, off, labels = C.load_raw()
nxt = np.r_[on[1:], len(cue)]
assert np.all(on + int(4.0 * fs) <= nxt), "a LOOSE/OPEN-AFTER window would run into the next cue"
P = {b: C.band_power(b, True) for b in C.BANDS}
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)   # behaviour only
W = {"REST": (-0.75, 0.0), "PAPER": (1.0, 1.75), "OPEN-AFTER": (3.25, 4.0), "LOOSE": (3.25, 4.0)}
CALIPER = 0.10          # max allowed Euclidean distance between the two windows' 5-finger posture
SPEED_CAL = 0.03        # max allowed difference in mean finger speed


def neural(o, w):
    return np.concatenate([C.win_log(P[b], o, *w) for b in C.BANDS])           # 5 bands x 60 ch


def posture(o, w):
    return glove[:, o + int(w[0] * fs): o + int(w[1] * fs)].mean(1)


def speed(o, w):
    seg = gs[:, o + int(w[0] * fs): o + int(w[1] * fs)]
    return float(np.sqrt((np.diff(seg, axis=1) ** 2).sum(0)).mean() * fs)


def motor_db(band, o, w):
    return float(10 * C.win_log(P[band], o, *w, C.MOTOR).mean())


def paired_cv(FA, FB, repeats=5, flip=None):
    """Balanced accuracy for telling window A from window B; both windows of a trial stay in the
    same fold. flip: boolean per pair, swaps the two labels for that pair (null)."""
    n = len(FA)
    X = np.r_[FA, FB]
    yy = np.r_[np.zeros(n, int), np.ones(n, int)]
    if flip is not None:
        yy = np.r_[np.where(flip, 1, 0), np.where(flip, 0, 1)]
    g = np.r_[np.arange(n), np.arange(n)]
    accs = []
    for r in range(repeats):
        perm = np.random.default_rng(1000 + r).permutation(n)            # different fold assignment per repeat
        gg = perm[g]
        pred = np.zeros(2 * n, int)
        for tr, te in GroupKFold(5).split(X, yy, gg):
            m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(X[tr], yy[tr])
            pred[te] = m.predict(X[te])
        accs.append(balanced_accuracy_score(yy, pred))
    return float(np.mean(accs))


def test(FA, FB, n_perm=200, repeats=3):
    real = paired_cv(FA, FB, repeats)
    null = np.array([paired_cv(FA, FB, repeats, flip=rng.random(len(FA)) < .5) for _ in range(n_perm)])
    return real, float(np.percentile(null, 95)), float((np.sum(null >= real) + 1) / (n_perm + 1))


comparisons = [("PAPER vs REST", "PAPER", "REST", labels == 3),
               ("PAPER vs OPEN-AFTER", "PAPER", "OPEN-AFTER", labels == 3),
               ("OPEN-AFTER vs REST", "OPEN-AFTER", "REST", labels == 3),
               ("LOOSE vs REST", "LOOSE", "REST", labels != 3)]
out = {}
for name, a, b, mask in comparisons:
    idx = np.where(mask)[0]
    pa = np.array([posture(on[i], W[a]) for i in idx]); pb = np.array([posture(on[i], W[b]) for i in idx])
    sa = np.array([speed(on[i], W[a]) for i in idx]); sb = np.array([speed(on[i], W[b]) for i in idx])
    dist = np.linalg.norm(pa - pb, axis=1)
    keep = (dist < CALIPER) & (np.abs(sa - sb) < SPEED_CAL)
    k = idx[keep]
    NA = np.array([neural(on[i], W[a]) for i in k]); NB = np.array([neural(on[i], W[b]) for i in k])
    GA = np.c_[pa[keep], sa[keep]]; GB = np.c_[pb[keep], sb[keep]]
    print(f"\n=== {name}: kept {len(k)}/{len(idx)} trial pairs (posture distance < {CALIPER}, speed diff < {SPEED_CAL})")
    print(f"  mean posture A: " + " ".join(f"{v:.2f}" for v in pa[keep].mean(0)) +
          f"   B: " + " ".join(f"{v:.2f}" for v in pb[keep].mean(0)) +
          f"   speed A {sa[keep].mean():.3f} / B {sb[keep].mean():.3f}")
    res = {"kept": int(len(k)), "of": int(len(idx)),
           "posture_A": pa[keep].mean(0).tolist(), "posture_B": pb[keep].mean(0).tolist(),
           "speed_A": float(sa[keep].mean()), "speed_B": float(sb[keep].mean())}
    if len(k) < 10:
        print("  too few matched pairs, skipped"); out[name] = res; continue
    for lab, FA, FB in [("glove (control: should be ~50%)", GA, GB), ("brain, all bands", NA, NB),
                        ("brain, high-gamma only", NA[:, 240:], NB[:, 240:]), ("brain, beta only", NA[:, 120:180], NB[:, 120:180])]:
        acc, n95, p = test(FA, FB)
        res[lab] = dict(acc=acc, null95=n95, p=p)
        print(f"  {lab:34s} {acc*100:5.1f}%  (pair-swap null 95th {n95*100:.1f}%, p={p:.3f})", flush=True)
    for band in ("highgamma", "beta"):
        va = np.array([motor_db(band, on[i], W[a]) for i in k]); vb = np.array([motor_db(band, on[i], W[b]) for i in k])
        w = stats.wilcoxon(va, vb)
        res[f"motor_{band}_diff_db"] = float((va - vb).mean()); res[f"motor_{band}_p"] = float(w.pvalue)
        res[f"_{band}"] = (va.tolist(), vb.tolist())
        print(f"  motor {band:9s} A - B = {(va-vb).mean():+.2f} dB (Wilcoxon p={w.pvalue:.4f})")
    out[name] = res

json.dump({k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in out.items()},
          open("results/same_hand.json", "w"), indent=2)

# ---------------- Figure ----------------
names = [c[0] for c in comparisons if "brain, all bands" in out[c[0]]]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
xs = np.arange(5)
cols = {"REST": "#8a8a85", "PAPER": "#1baf7a", "OPEN-AFTER": "#7a5cc4", "LOOSE": "#eb6834"}
shown = set()
for name, a, b, mask in comparisons:
    r = out[name]
    for cond, key in [(a, "posture_A"), (b, "posture_B")]:
        if cond in shown:
            continue
        shown.add(cond)
        ax[0].plot(xs, r[key], "-o", color=cols[cond], lw=2, label=cond)
ax[0].set_xticks(xs, C.FINGERS)
ax[0].set(ylim=(0, 0.8), ylabel="Finger bend (glove)", title="A. Matched hand posture in all four conditions")
ax[0].legend(frameon=False)
w = 0.38
for j, (lab, col) in enumerate([("glove (control: should be ~50%)", "#c3c2b7"), ("brain, all bands", "#2a78d6")]):
    vals = [out[n][lab]["acc"] * 100 for n in names]
    ax[1].bar(np.arange(len(names)) + (j - .5) * w, vals, w - .03, color=col, label="glove (posture + speed)" if j == 0 else "brain")
    for i, n in enumerate(names):
        x = i + (j - .5) * w
        ax[1].plot([x - w / 2, x + w / 2], [out[n][lab]["null95"] * 100] * 2, "k--", lw=1)
        ax[1].text(x, vals[i] + 1, f"p={out[n][lab]['p']:.3f}", ha="center", fontsize=7, rotation=90, va="bottom")
ax[1].axhline(50, color="k", lw=.6)
ax[1].set_xticks(range(len(names)), [n.replace(" vs ", "\nvs ") + f"\n(n={out[n]['kept']})" for n in names], fontsize=8)
ax[1].set(ylim=(30, 105), ylabel="Balanced accuracy (%)", title="B. Can we tell the two windows apart?\n(dashed = pair-swap null 95th)")
ax[1].legend(frameon=False, loc="upper left")
for i, n in enumerate(names):
    for j, (band, col) in enumerate([("highgamma", "#2a78d6"), ("beta", "#eda100")]):
        ax[2].bar(i + (j - .5) * w, out[n][f"motor_{band}_diff_db"], w - .03, color=col, label=("motor high-gamma" if band == "highgamma" else "motor beta") if i == 0 else None)
        ax[2].text(i + (j - .5) * w, out[n][f"motor_{band}_diff_db"], f"p={out[n][f'motor_{band}_p']:.3f}", ha="center", fontsize=7,
                   va="bottom" if out[n][f"motor_{band}_diff_db"] >= 0 else "top")
ax[2].axhline(0, color="k", lw=.6)
ax[2].set_xticks(range(len(names)), [n.replace(" vs ", "\nvs ") for n in names], fontsize=8)
ax[2].set(ylabel="First minus second (dB)", title="C. Motor-channel power difference")
ax[2].legend(frameon=False, fontsize=8)
for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/E1_same_hand.png", dpi=160)
