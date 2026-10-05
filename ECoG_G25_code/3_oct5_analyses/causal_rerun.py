"""
Test 1 (Oct 5): rerun the baseline / carryover claims with CAUSAL filtering.

Why: the original pipeline filters forward AND backward (zero-phase). The
backward pass lets a little of what happens AFTER the cue leak into the
"last 1 s before the cue". A causal (forward-only) filter cannot do that.
If a pre-cue claim survives the causal version, it is not a filter artefact.

Every claim is computed twice, side by side: ZERO-PHASE (original) and CAUSAL.
A third "guard" column uses zero-phase but ends the pre-cue window 0.25 s
before the cue, a second, independent way of keeping post-cue activity out.

Claims tested
  A  Pre-cue motor beta: after a movement vs after open hand (+1.63 dB, p=0.0018)
  B  Decode "previous trial was a movement" from the pre-cue state (75%)
  C  Decode previous FIST vs PEACE from the pre-cue state (55%, at chance)
  D  Carryover: movement response smaller after a movement (-0.44 dB, p=0.004)
  E  Pre-cue motor high-gamma by previous gesture
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import common as C

N_PERM = 200
rng = np.random.default_rng(0)
y, cue, glove, on, off, labels = C.load_raw()
prev = np.r_[0, labels[:-1]]
has_prev = np.arange(90) > 0


def cv_bal(F, yy, seeds=(0, 1, 2)):
    m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    return np.mean([cross_val_score(m, F, yy, scoring="balanced_accuracy",
                                    cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean() for s in seeds])


def perm(F, yy, n=N_PERM):
    real = cv_bal(F, yy)
    null = np.array([cv_bal(F, rng.permutation(yy), seeds=(0,)) for _ in range(n)])
    return real, null, (np.sum(null >= real) + 1) / (n + 1)


def run(causal, end=0.0):
    P = {b: C.band_power(b, causal) for b in C.BANDS}
    t0, t1 = end - 1.0, end
    out = {}
    # A: pre-cue motor beta
    beta = np.array([10 * C.win_log(P["beta"], o, t0, t1, C.MOTOR).mean() for o in on])
    mv, op = has_prev & (prev != 3), prev == 3
    u = stats.mannwhitneyu(beta[mv], beta[op])
    out["A_beta_diff_db"] = beta[mv].mean() - beta[op].mean()
    out["A_p"] = u.pvalue
    out["A_by_prev"] = {n: float(beta[prev == k].mean()) for k, n in [(1, "fist"), (2, "peace"), (3, "open")]}
    out["_beta"] = beta
    # B: decode previous moved (balanced accuracy, chance 50%)
    B = np.array([np.concatenate([C.win_log(P[b], o, t0, t1) for b in C.BANDS]) for o in on])
    r, null, p = perm(B[has_prev], (prev[has_prev] != 3).astype(int))
    out["B_bal_acc"], out["B_null95"], out["B_p"] = r, np.percentile(null, 95), p
    # C: previous fist vs peace
    fp = (prev == 1) | (prev == 2)
    r, null, p = perm(B[fp], prev[fp])
    out["C_bal_acc"], out["C_null95"], out["C_p"] = r, np.percentile(null, 95), p
    # E: pre-cue motor high-gamma by previous gesture
    hgpre = np.array([10 * C.win_log(P["highgamma"], o, t0, t1, C.MOTOR).mean() for o in on])
    out["E_hg_by_prev"] = {n: float(hgpre[prev == k].mean()) for k, n in [(1, "fist"), (2, "peace"), (3, "open")]}
    out["E_hg_mv_vs_open_p"] = stats.mannwhitneyu(hgpre[mv], hgpre[op]).pvalue
    out["_hgpre"] = hgpre
    # D: carryover response size (top 5 channels, 0.4-1.2 s), only meaningful with end=0
    hg = P["highgamma"]
    top = [25, 36, 26, 15, 46]
    size = np.array([10 * np.log10(hg[top, o + int(0.4 * C.fs): o + int(1.2 * C.fs)].mean()) for o in on])
    pre = np.array([10 * np.log10(hg[top, o - int(0.5 * C.fs): o].mean()) for o in on])
    m = has_prev & (labels != 3)
    for name, v in [("D_abs", size), ("D_change", size - pre)]:
        X = np.c_[np.ones(m.sum()), prev[m] != 3, labels[m] == 2, np.arange(90)[m] / 90]
        coef = np.linalg.lstsq(X, v[m], rcond=None)[0]
        res = v[m] - X @ coef
        se = np.sqrt(np.diag(res @ res / (m.sum() - 4) * np.linalg.inv(X.T @ X)))
        out[name + "_coef"] = coef[1]
        out[name + "_p"] = 2 * stats.t.sf(abs(coef[1] / se[1]), m.sum() - 4)
    out["_size"], out["_pre"] = size, pre
    return out


res = {"zero-phase (original)": run(False), "causal": run(True), "zero-phase, window ends -0.25 s": run(False, -0.25)}

rows = [("A  pre-cue motor beta, after movement - after open (dB)", "A_beta_diff_db", "A_p"),
        ("B  decode 'previous moved' (balanced acc, chance 50%)", "B_bal_acc", "B_p"),
        ("C  decode previous fist vs peace (balanced acc, chance 50%)", "C_bal_acc", "C_p"),
        ("D  carryover, absolute response size (dB)", "D_abs_coef", "D_abs_p"),
        ("D' carryover, response change from own pre-cue (dB)", "D_change_coef", "D_change_p")]
print(f"{'':62s}" + "".join(f"{k:>34s}" for k in res))
for title, v, p in rows:
    print(f"{title:62s}" + "".join(f"{r[v]:>22.3f}  (p={r[p]:.4f})" for r in res.values()))
for k, r in res.items():
    print(f"\n[{k}] B shuffled 95th = {r['B_null95']:.3f}; C shuffled 95th = {r['C_null95']:.3f}")
    print(f"  pre-cue beta by previous gesture (dB): {r['A_by_prev']}")
    print(f"  pre-cue HIGH-GAMMA by previous gesture (dB): {r['E_hg_by_prev']}  (moved vs open p={r['E_hg_mv_vs_open_p']:.4f})")

json.dump({k: {a: (b if not isinstance(b, np.ndarray) else None) for a, b in r.items() if not a.startswith("_")}
           for k, r in res.items()}, open("results/causal_rerun.json", "w"), indent=2, default=float)
np.savez("results/causal_trial_values.npz", beta_causal=res["causal"]["_beta"], beta_zp=res["zero-phase (original)"]["_beta"],
         hgpre_causal=res["causal"]["_hgpre"], size_causal=res["causal"]["_size"], pre_causal=res["causal"]["_pre"])

# ---------------- Figure ----------------
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
for i, (k, col) in enumerate(zip(["zero-phase (original)", "causal"], [GREY, BLUE])):
    b = res[k]["_beta"]
    for j, (mask, name) in enumerate([(has_prev & (prev != 3), "after fist/peace"), (prev == 3, "after open")]):
        x = j * 3 + i
        ax[0].boxplot([b[mask]], positions=[x], widths=0.7, showfliers=False, medianprops=dict(color="k"))
        ax[0].scatter(np.full(mask.sum(), x) + rng.uniform(-.15, .15, mask.sum()), b[mask], s=12, color=col, alpha=.7)
ax[0].set_xticks([0.5, 3.5], ["After fist / peace", "After open"])
ax[0].set(ylabel="Motor beta, last 1 s before cue (dB)",
          title=f"A. Pre-cue beta: zero-phase (grey) vs causal (blue)\n"
                f"causal diff {res['causal']['A_beta_diff_db']:+.2f} dB, p={res['causal']['A_p']:.4f}")
labels_k = ["Zero-phase", "Causal", "ZP, ends -0.25 s"]
cols = [GREY, BLUE, "#c3c2b7"]
w = 0.26
for i, (k, r) in enumerate(res.items()):
    vals = [r["B_bal_acc"] * 100, r["C_bal_acc"] * 100]
    nulls = [r["B_null95"] * 100, r["C_null95"] * 100]
    xs = np.arange(2) + (i - 1) * w
    ax[1].bar(xs, vals, w - .03, color=cols[i], label=labels_k[i])
    for x, n in zip(xs, nulls):
        ax[1].plot([x - w / 2, x + w / 2], [n, n], color="k", ls="--", lw=1.2)
ax[1].axhline(50, color="k", lw=.6)
ax[1].set_xticks([0, 1], ["Previous trial moved?", "Previous fist or peace?"])
ax[1].set(ylabel="Balanced accuracy (%)", ylim=(30, 100), title="B/C. Decoding the past from the pre-cue state\n(dashed = shuffled-label 95th pct)")
ax[1].legend(frameon=False, loc="upper right")
for i, (k, r) in enumerate(res.items()):
    xs = np.arange(2) + (i - 1) * w
    ax[2].bar(xs, [r["D_abs_coef"], r["D_change_coef"]], w - .03, color=cols[i], label=labels_k[i])
    for x, p in zip(xs, [r["D_abs_p"], r["D_change_p"]]):
        ax[2].text(x, -0.03, f"p={p:.3f}", rotation=90, ha="center", va="top", fontsize=8, color="white")
ax[2].axhline(0, color="k", lw=.6)
ax[2].set_xticks([0, 1], ["Absolute response size", "Change from own pre-cue"])
ax[2].set(ylabel="After movement - after open (dB)", title="D. Carryover: smaller response after a movement\n(controls current gesture + trial number)")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/T1_causal_rerun.png", dpi=160)
