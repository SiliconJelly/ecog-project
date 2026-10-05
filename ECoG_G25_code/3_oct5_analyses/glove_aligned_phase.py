"""
Test 4 (Oct 5): align to the HAND, not the screen, and repeat the phase transfer.

Cue-aligned windows mix trials where the hand moved early or late. Here each
fist/peace trial is aligned to when the glove shows the fingers actually start
to move (movement onset) and actually start to open again (release onset).

Then the same question as Oct 4: does a decoder trained while the hand CLOSES
work while it OPENS (and vice versa)?

Windows (equal length, 7 x 100 ms bins):
  make    = movement onset + 0.0 ... 0.7 s
  release = release onset  + 0.0 ... 0.7 s
Same test with the cue-aligned equal windows is printed for comparison.
Phase-swap permutation (see phase_permutation.py) tests the transfer deficit.
Neural features: causal filtering.
"""
import json, sys
import numpy as np
import scipy.signal as ss
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import common as C

fs, step = C.fs, 0.1
N_SWAP = int(sys.argv[1]) if len(sys.argv) > 1 else 500
rng = np.random.default_rng(0)
y, cue, glove, on, off, labels = C.load_raw()
fp = np.where(labels != 3)[0]

# ---------------- glove onsets ----------------
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)   # behavioural smoothing only
move_on, rel_on = np.full(90, np.nan), np.full(90, np.nan)
for i in range(90):
    o, f = on[i], off[i]
    base = gs[:, o - fs:o].mean(1)
    d = np.sqrt(((gs[:, o:o + 2 * fs] - base[:, None]) ** 2).mean(0))
    a = np.percentile(d, 95)
    h = np.where(d > max(.02, .3 * a))[0]
    if len(h):
        move_on[i] = h[0] / fs                                   # s after cue
    holdp = gs[:, o + int(.75 * fs):o + int(1.75 * fs)].mean(1)
    seg = gs[:, o + int(1.75 * fs): f + int(1.5 * fs)]
    d2 = np.sqrt(((seg - holdp[:, None]) ** 2).mean(0))
    a2 = np.percentile(d2, 95)
    h2 = np.where(d2 > max(.02, .3 * a2))[0]
    if len(h2):
        rel_on[i] = 1.75 + h2[0] / fs                            # s after cue
ok = fp[np.isfinite(move_on[fp]) & np.isfinite(rel_on[fp])]
print(f"Fist/peace trials with both onsets: {len(ok)}/{len(fp)}")
for k, n in [(1, "fist"), (2, "peace")]:
    m = ok[labels[ok] == k]
    print(f"  {n:5s}: movement onset {np.median(move_on[m]):.2f} s (IQR {np.percentile(move_on[m],25):.2f}-{np.percentile(move_on[m],75):.2f}),"
          f" release onset {np.median(rel_on[m]):.2f} s after cue (cue off at 2.0 s;"
          f" IQR {np.percentile(rel_on[m],25):.2f}-{np.percentile(rel_on[m],75):.2f})")
out = dict(n_trials=int(len(ok)),
           move_onset=dict(median=float(np.median(move_on[ok])), sd=float(np.std(move_on[ok]))),
           release_onset=dict(median=float(np.median(rel_on[ok])), sd=float(np.std(rel_on[ok]))))
print(f"  Spread (SD) of onsets across trials: movement {np.std(move_on[ok])*1000:.0f} ms, release {np.std(rel_on[ok])*1000:.0f} ms")

# ---------------- features ----------------
hg = C.band_power(None, True, lo_hi=(52, 300))
rel_t = np.arange(-0.5, 1.2, step)                                 # time around each event


def feats(event_s):
    return np.stack([np.log10(np.stack([hg[:, on[i] + int((e + t) * fs): on[i] + int((e + t + step) * fs)].mean(axis=1)
                                        for t in rel_t], axis=1)) for i, e in zip(ok, event_s)])


Xm, Xr = feats(move_on[ok]), feats(rel_on[ok])                     # trials x ch x bins
Xc_m, Xc_r = feats(np.full(len(ok), np.median(move_on[ok]))), feats(np.full(len(ok), np.median(rel_on[ok])))   # cue-aligned: same window for every trial, placed at the median onset
yy = labels[ok]
W = np.where((rel_t >= -1e-9) & (rel_t < 0.7 - 1e-9))[0]            # 0..0.7 s after event (7 bins)


def blocks(XA, XB, yy, repeats=3):
    res = dict(make_make=0., release_release=0., make_release=0., release_make=0.)
    for s in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(XA[:, :, 0], yy):
            for i in W:
                for src, name_s in [(XA, "make"), (XB, "release")]:
                    m = LDA(solver="lsqr", shrinkage="auto").fit(src[tr, :, i], yy[tr])
                    for j in W:
                        for tgt, name_t in [(XA, "make"), (XB, "release")]:
                            res[f"{name_s}_{name_t}"] += (m.predict(tgt[te, :, j]) == yy[te]).sum()
    return {k: v / (len(yy) * repeats * len(W) ** 2) for k, v in res.items()}


def deficit(b):
    return (b["make_make"] + b["release_release"]) / 2 - (b["make_release"] + b["release_make"]) / 2


summary = {}
for name, XA, XB in [("glove-aligned", Xm, Xr), ("cue-aligned (shifted by median latency)", Xc_m, Xc_r)]:
    b = blocks(XA, XB, yy)
    d = deficit(b)
    null = []
    for _ in range(N_SWAP):
        sw = rng.random(len(yy)) < .5
        A2, B2 = XA.copy(), XB.copy()
        A2[sw], B2[sw] = XB[sw], XA[sw]
        null.append(deficit(blocks(A2, B2, yy, repeats=1)))
    null = np.array(null)
    p = (np.sum(null >= d) + 1) / (len(null) + 1)
    summary[name] = dict(blocks=b, deficit=d, null_mean=float(null.mean()), null95=float(np.percentile(null, 95)), p=p)
    print(f"\n[{name}]  fist vs peace, chance 50%")
    for k, v in b.items():
        print(f"  {k:16s} {v*100:5.1f}%")
    print(f"  Transfer deficit {d*100:+.1f} pts  (phase-swap null mean {null.mean()*100:+.1f}, 95th {np.percentile(null,95)*100:+.1f}; p={p:.3f})")
    sys.stdout.flush()
out["summary"] = summary

# Full generalization matrix across both aligned segments, for the figure
seg = np.concatenate([Xm, Xr], axis=2)
nb = seg.shape[2]
M = np.zeros((nb, nb))
for s in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(seg[:, :, 0], yy):
        for i in range(nb):
            m = LDA(solver="lsqr", shrinkage="auto").fit(seg[tr, :, i], yy[tr])
            for j in range(nb):
                M[i, j] += (m.predict(seg[te, :, j]) == yy[te]).sum()
M /= len(yy) * 3
np.save("results/glove_aligned_tg.npy", M)
json.dump(out, open("results/glove_aligned_phase.json", "w"), indent=2, default=float)

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25, 1]})
for k, col, n in [(1, "#2a78d6", "Fist"), (2, "#eb6834", "Peace")]:
    m = labels[ok] == k
    ax[0].scatter(move_on[ok][m], rel_on[ok][m], color=col, s=25, alpha=.8, label=n)
ax[0].axhline(2.0, color="k", ls="--", lw=.8)
ax[0].text(0.05, 2.02, "cue off", fontsize=8, transform=ax[0].get_yaxis_transform())
ax[0].set(xlabel="Movement onset (s after cue)", ylabel="Release onset (s after cue)", title="A. When the hand actually moved (glove)")
ax[0].legend(frameon=False)
norm = TwoSlopeNorm(vcenter=.5, vmin=0.2, vmax=1)
im = ax[1].imshow(M, origin="lower", cmap="RdBu_r", norm=norm, extent=[0, nb, 0, nb])
for v in (nb / 2,):
    ax[1].axvline(v, color="k", lw=1.2)
    ax[1].axhline(v, color="k", lw=1.2)
ticks = [np.argmin(abs(rel_t - t)) for t in (0, 0.5, 1.0)]
tk = ticks + [t + len(rel_t) for t in ticks]
ax[1].set_xticks(np.array(tk) + .5, ["0", ".5", "1"] * 2)
ax[1].set_yticks(np.array(tk) + .5, ["0", ".5", "1"] * 2)
ax[1].set(xlabel="Test time: [ s from MOVEMENT onset | s from RELEASE onset ]",
          ylabel="Train time: [ movement | release ]", title="B. Fist vs peace, glove-aligned generalization\n(white = chance; off-diagonal squares = transfer)")
fig.colorbar(im, ax=ax[1], shrink=.8, label="Accuracy")
order = ["make_make", "release_release", "make_release", "release_make"]
for j, (name, col) in enumerate(zip(summary, ["#2a78d6", "#8a8a85"])):
    xs = np.arange(4) + (j - .5) * .38
    ax[2].bar(xs, [summary[name]["blocks"][k] * 100 for k in order], .35, color=col, label=name.split(" (")[0])
ax[2].axhline(50, color="k", lw=.6)
ax[2].set_xticks(range(4), ["make→make", "release→release", "make→release", "release→make"], fontsize=8)
ax[2].set(ylim=(40, 80), ylabel="Accuracy (%)",
          title="C. Glove- vs cue-aligned\n" + "; ".join(f"{n.split(' (')[0]} deficit p={summary[n]['p']:.3f}" for n in summary))
ax[2].legend(frameon=False)
for a in (ax[0], ax[2]):
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/T4_glove_aligned_phase.png", dpi=160)
