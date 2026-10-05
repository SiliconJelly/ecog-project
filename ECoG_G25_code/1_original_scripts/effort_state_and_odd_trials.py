"""
Two checks on the decoder's peace <-> open mix-ups.

Part 1  Effort-state feature: add ONE extra input (pre-cue motor beta, the
        "I just moved" state) to the standard decoder. Does it reduce errors?
        Win criterion (decided BEFORE running): total errors must drop by more
        than they do when the same feature is shuffled across trials, without
        just moving errors to other cells.
Part 2  Which trials cause the peace <-> open errors, and what did the hand
        (glove) and brain actually do on those trials compared to typical
        open and peace trials?

Takes about 3-4 minutes.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

fs = 1200
rng = np.random.default_rng(0)
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
FCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# ---------------- Load, clean, band power ----------------
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog, cue, glove = y[1:61], y[61], y[62:67]
ecog = ecog - ecog.mean(axis=0)
ecog = ss.sosfiltfilt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), ecog, axis=1)
for f0 in [50, 100, 150, 200, 250, 300]:
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)
hg = ss.sosfiltfilt(ss.butter(4, [52, 300], "bandpass", fs=fs, output="sos"), ecog, axis=1) ** 2
beta = ss.sosfiltfilt(ss.butter(4, [13, 30], "bandpass", fs=fs, output="sos"), ecog, axis=1) ** 2
print("Filtering done.")

onsets = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
offsets = np.where((np.diff(cue) != 0) & (cue[1:] == 0))[0] + 1
labels = cue[onsets].astype(int)                 # 1 fist, 2 peace, 3 open
prev = np.r_[0, labels[:-1]]
motor = [15, 25, 26, 36, 46]

# Standard features (same as classify.py) + the one extra effort-state feature
edges = np.arange(0.25, 2.26, 0.25)
X = np.array([np.concatenate([np.log10(hg[:, o + int(a * fs): o + int(b * fs)].mean(axis=1))
                              for a, b in zip(edges[:-1], edges[1:])]) for o in onsets])
effort = np.array([10 * np.log10(beta[motor, o - fs: o].mean()) for o in onsets])

def run(F, runs=20):
    """Average confusion matrix + how often each trial is wrong, over repeated 10-fold CV."""
    M, wrong = np.zeros((3, 3)), np.zeros(len(labels))
    model = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    for s in range(runs):
        pred = cross_val_predict(model, F, labels, cv=StratifiedKFold(10, shuffle=True, random_state=s))
        M += confusion_matrix(labels, pred, labels=[1, 2, 3])
        wrong += pred != labels
    return M / runs, wrong / runs

def errors(M):
    return dict(total=M.sum() - np.trace(M), peace_open=M[1, 2] + M[2, 1],
                fist_peace=M[0, 1] + M[1, 0], fist_open=M[0, 2] + M[2, 0])

# ---------------- Part 1: effort-state feature ----------------
M_std, wrong = run(X)
M_eff, _ = run(np.c_[X, effort])
e_std, e_eff = errors(M_std), errors(M_eff)

null_change = []
for _ in range(40):                                  # same feature, shuffled across trials
    M_shuf, _ = run(np.c_[X, rng.permutation(effort)], runs=5)
    null_change.append(errors(M_shuf)["total"] - e_std["total"])
null_change = np.array(null_change)
real_change = e_eff["total"] - e_std["total"]

print("\nPart 1: does the pre-cue effort state reduce errors? (average per run of 90 trials)")
print(f"  {'':22s} {'total':>6s} {'peace<->open':>13s} {'fist<->peace':>13s} {'accuracy':>9s}")
for name, e, M in [("Standard", e_std, M_std), ("+ effort state", e_eff, M_eff)]:
    print(f"  {name:22s} {e['total']:6.2f} {e['peace_open']:13.2f} {e['fist_peace']:13.2f} {np.trace(M)/90*100:8.1f}%")
print(f"  Change in total errors: {real_change:+.2f}   (shuffled feature: mean {null_change.mean():+.2f}, "
      f"5th pct {np.percentile(null_change, 5):+.2f})")
win = real_change < np.percentile(null_change, 5)
print(f"  Verdict: {'WIN - beats the shuffled feature' if win else 'NO EFFECT - no better than a shuffled feature'}")

# ---------------- Part 2: the odd trials ----------------
odd = np.where((labels == 3) & (wrong >= 0.25))[0]   # open trials misread in >= 25% of runs
gaps = np.r_[np.nan, (onsets[1:] - offsets[:-1]) / fs]
print("\nPart 2: open-hand trials the decoder keeps getting wrong")
for i in odd:
    p = {0: "none (first trial)", 1: "fist", 2: "peace", 3: "open"}[prev[i]]
    gap = "n/a" if np.isnan(gaps[i]) else f"{gaps[i]:.2f} s (median {np.nanmedian(gaps):.2f})"
    print(f"  Trial {i+1}: wrong in {wrong[i]*100:.0f}% of runs, previous trial = {p}, rest gap = {gap}")

tt = np.arange(-0.5, 3.0, 0.05)
step = int(0.05 * fs)
G = np.array([[glove[:, o + int(t * fs): o + int(t * fs) + step].mean(axis=1) for t in tt] for o in onsets])
H = np.array([[10 * np.log10(hg[motor, o + int(t * fs): o + int(t * fs) + step].mean()) for t in tt] for o in onsets])
H -= H[:, tt < 0].mean(axis=1, keepdims=True)
hold = (tt > 0.75) & (tt < 2.0)

typical = {}
for k, name in [(2, "peace"), (3, "open")]:
    m = labels == k
    m[odd] = False
    typical[name] = m
print("\n  Finger bend during the hold (0 = straight, 1 = fully bent) and brain response:")
print(f"  {'':16s}" + "".join(f"{f:>8s}" for f in FINGERS) + "   high-gamma")
for name, rows in [("Typical peace", G[typical["peace"]].mean(axis=0)),
                   ("Typical open", G[typical["open"]].mean(axis=0))] + \
                  [(f"Trial {i+1}", G[i]) for i in odd]:
    hgv = (H[typical[name.split()[1]]].mean(axis=0) if name.startswith("Typical") else H[int(name.split()[1]) - 1])[hold].mean()
    print(f"  {name:16s}" + "".join(f"{v:8.2f}" for v in rows[hold].mean(axis=0)) + f"   {hgv:+.2f} dB")
for i in odd:
    print(f"  Trial {i+1} BEFORE the cue:" + "".join(f"{v:8.2f}" for v in G[i][tt < 0].mean(axis=0)))

# ---------------- Figure ----------------
panels = [("Typical open", G[typical["open"]].mean(axis=0)), ("Typical peace", G[typical["peace"]].mean(axis=0))] + \
         [(f"Trial {i+1} (open, misread)", G[i]) for i in odd]
fig, ax = plt.subplots(1, len(panels) + 1, figsize=(4 * (len(panels) + 1), 4), layout="constrained")
for a, (title, g) in zip(ax, panels):
    for f in range(5):
        a.plot(tt, g[:, f], color=FCOL[f], lw=2, label=FINGERS[f])
    a.axvspan(0, 2, color="grey", alpha=0.1)
    a.set(title=title, xlabel="Time from cue (s)", ylim=(-0.05, 1.05))
ax[0].set_ylabel("Finger bend (glove)")
ax[0].legend(frameon=False, fontsize=8, loc="upper right")

a = ax[-1]
a.plot(tt, H[typical["open"]].mean(axis=0), color="#1baf7a", lw=2, label="Typical open")
a.plot(tt, H[typical["peace"]].mean(axis=0), color="#eb6834", lw=2, label="Typical peace")
for i, ls in zip(odd, ["-", "--", ":"]):
    a.plot(tt, H[i], color="k", lw=1.5, ls=ls, label=f"Trial {i+1}")
a.axvspan(0, 2, color="grey", alpha=0.1)
a.axhline(0, color="k", lw=0.5)
a.set(title="Brain: motor high-gamma", xlabel="Time from cue (s)", ylabel="Change from pre-cue (dB)")
a.legend(frameon=False, fontsize=8)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.show()
