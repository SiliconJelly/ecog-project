"""
Test 3 (Oct 5): permutation tests for the MAKE vs RELEASE phase contrasts.

Oct 4 (temporal_generalization.py) found, fist vs peace:
   make->make 61.3%, release->release 60.9%, make->release 49.4%, release->make 52.6%
but no significance test. Two questions, two different nulls:

Q1  Is each block above chance?            Null = shuffle the gesture labels.
Q2  Is cross-phase transfer WORSE than      Null = for each trial, randomly swap its
    within-phase decoding?                  make and release data ("phase swap").
                                            If the code were the same in both phases,
                                            swapping them would change nothing.
    Q2 needs equal-length windows, so it uses make 0.5-1.2 s and release 2.3-3.0 s (7 bins each).

Run for the original zero-phase features (exact Oct 4 numbers) and for causal features.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import common as C

fs, step = C.fs, 0.1
times = np.arange(-1, 4.0, step)              # identical to temporal_generalization.py
N_LABEL, N_SWAP = int(sys.argv[1]) if len(sys.argv) > 1 else 500, int(sys.argv[2]) if len(sys.argv) > 2 else 500
rng = np.random.default_rng(0)
y, cue, glove, on, off, labels = C.load_raw()


def tg_features(causal, event_samples):
    hg = C.band_power(None, causal, lo_hi=(52, 300))
    return np.stack([np.log10(np.stack([hg[:, o + int(t * fs): o + int((t + step) * fs)].mean(axis=1)
                                        for t in times], axis=1)) for o in event_samples])   # trials x ch x bins


def sel(win):
    return np.where((times >= win[0]) & (times < win[1]))[0]      # same selection as the original block()


def blocks(X, yy, A, B, repeats=3):
    """Mean accuracy in the 4 blocks of the temporal-generalization matrix, restricted to bins A (make) and B (release)."""
    idx = np.r_[A, B]
    M = np.zeros((len(idx), len(idx)))
    for seed in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[:, :, 0], yy):
            for a, i in enumerate(idx):
                m = LDA(solver="lsqr", shrinkage="auto").fit(X[tr, :, i], yy[tr])
                for b, j in enumerate(idx):
                    M[a, b] += (m.predict(X[te, :, j]) == yy[te]).sum()
    M /= len(yy) * repeats
    nA = len(A)
    return dict(make_make=M[:nA, :nA].mean(), release_release=M[nA:, nA:].mean(),
                make_release=M[:nA, nA:].mean(), release_make=M[nA:, :nA].mean())


def deficit(b):
    return (b["make_make"] + b["release_release"]) / 2 - (b["make_release"] + b["release_make"]) / 2


MAKE, RELEASE, MAKE_EQ = (0.5, 1.5), (2.3, 3.0), (0.5, 1.2)
results = {}
for causal in (False, True):
    tag = "causal" if causal else "zero-phase"
    Xall = tg_features(causal, on)
    for task, keep, chance in [("3-class", labels > 0, 1 / 3), ("fist-vs-peace", labels != 3, 0.5)]:
        X, yy = Xall[keep], labels[keep]
        A, B = sel(MAKE), sel(RELEASE)
        real = blocks(X, yy, A, B)
        null = [blocks(X, rng.permutation(yy), A, B, repeats=1) for _ in range(N_LABEL)]
        p_above, p_two = {}, {}
        for k in real:
            nv = np.array([n[k] for n in null])
            p_above[k] = (np.sum(nv >= real[k]) + 1) / (len(nv) + 1)
            p_two[k] = (np.sum(np.abs(nv - nv.mean()) >= abs(real[k] - nv.mean())) + 1) / (len(nv) + 1)
        null95 = {k: float(np.percentile([n[k] for n in null], 95)) for k in real}
        null05 = {k: float(np.percentile([n[k] for n in null], 5)) for k in real}
        # Q2 phase swap, equal windows
        Ae, Be = sel(MAKE_EQ), sel(RELEASE)
        real_eq = blocks(X, yy, Ae, Be)
        d_real = deficit(real_eq)
        d_null = []
        for _ in range(N_SWAP):
            Xs = X.copy()
            sw = rng.random(len(yy)) < 0.5
            Xs[np.ix_(sw, np.arange(X.shape[1]), Ae)] = X[np.ix_(sw, np.arange(X.shape[1]), Be)]
            Xs[np.ix_(sw, np.arange(X.shape[1]), Be)] = X[np.ix_(sw, np.arange(X.shape[1]), Ae)]
            d_null.append(deficit(blocks(Xs, yy, Ae, Be, repeats=1)))
        d_null = np.array(d_null)
        p_def = (np.sum(d_null >= d_real) + 1) / (len(d_null) + 1)
        key = f"{tag} | {task}"
        results[key] = dict(chance=chance, blocks=real, p_above_chance=p_above, p_two_sided=p_two,
                            null95=null95, null05=null05, equal_windows=real_eq, deficit=d_real,
                            deficit_null_mean=float(d_null.mean()), deficit_null95=float(np.percentile(d_null, 95)),
                            p_deficit=p_def, _d_null=d_null.tolist())
        print(f"\n[{key}]  chance {chance*100:.0f}%")
        for k in real:
            print(f"  {k:16s} {real[k]*100:5.1f}%   shuffled 5-95%: {null05[k]*100:4.1f}-{null95[k]*100:4.1f}%   p(above)={p_above[k]:.3f}  p(two-sided)={p_two[k]:.3f}")
        print(f"  Transfer deficit (equal windows): {d_real*100:+.1f} pts; phase-swap null mean {d_null.mean()*100:+.1f}, "
              f"95th {np.percentile(d_null,95)*100:+.1f}; p={p_def:.3f}")
        sys.stdout.flush()

json.dump({k: {a: b for a, b in v.items()} for k, v in results.items()}, open("results/phase_permutation.json", "w"), indent=2, default=float)

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
order = ["make_make", "release_release", "make_release", "release_make"]
nice = ["make→make", "release→release", "make→release", "release→make"]
for a, task in zip(ax[:2], ["3-class", "fist-vs-peace"]):
    for j, (tag, col) in enumerate([("zero-phase", "#8a8a85"), ("causal", "#2a78d6")]):
        r = results[f"{tag} | {task}"]
        xs = np.arange(4) + (j - .5) * .38
        a.bar(xs, [r["blocks"][k] * 100 for k in order], .35, color=col, label=tag)
        for x, k in zip(xs, order):
            a.plot([x - .17, x + .17], [r["null95"][k] * 100] * 2, "k--", lw=1)
            a.text(x, r["blocks"][k] * 100 + 1, f"p={r['p_above_chance'][k]:.3f}", ha="center", fontsize=7, rotation=90, va="bottom")
    a.axhline(r["chance"] * 100, color="k", lw=.6)
    a.set_xticks(range(4), nice, fontsize=8)
    a.set(ylabel="Accuracy (%)", ylim=(25, 85), title=f"{task}: each block vs shuffled labels\n(dashed = shuffle 95th pct)")
    a.legend(frameon=False, loc="upper right")
for j, (task, col) in enumerate([("3-class", "#7a5cc4"), ("fist-vs-peace", "#eb6834")]):
    r = results[f"causal | {task}"]
    ax[2].hist(np.array(r["_d_null"]) * 100, bins=25, color=col, alpha=.45, label=f"{task} null (phase swapped)")
    ax[2].axvline(r["deficit"] * 100, color=col, lw=2.5, label=f"{task} real: {r['deficit']*100:+.1f} pts, p={r['p_deficit']:.3f}")
ax[2].set(xlabel="Within-phase minus cross-phase accuracy (pts)", ylabel="Count",
          title="Is transfer between phases worse than chance\nwould allow if the code were shared? (causal)")
ax[2].legend(frameon=False, fontsize=8)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/T3_phase_permutation.png", dpi=160)
