"""
Temporal generalization: is the brain's "code" for a gesture the same over time?

Train a decoder at one moment (train time), test it at every other moment
(test time). Result = a matrix:
  - diagonal  = normal decoding at each moment
  - off-diagonal high  -> the same brain pattern is reused at both times
  - off-diagonal at chance -> the code has CHANGED between those times

Run twice: all 3 gestures, and fist vs peace only (removes the trivial
"moved vs didn't move" difference, since open hand barely moves).

Takes about 3-4 minutes.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

fs = 1200
step = 0.1                                    # 100 ms time bins
times = np.arange(-1, 4.0, step)              # -1 s to +4 s around the cue

# ---- Load, clean, high-gamma power (same as before) ----
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog, cue = y[1:61], y[61]
ecog = ecog - ecog.mean(axis=0)
ecog = ss.sosfiltfilt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), ecog, axis=1)
for f0 in [50, 100, 150, 200, 250, 300]:
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)
hg = ss.sosfiltfilt(ss.butter(4, [52, 300], "bandpass", fs=fs, output="sos"), ecog, axis=1) ** 2

onsets = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
labels = cue[onsets].astype(int)

# trials x channels x time bins
X = np.stack([np.log10(np.stack([hg[:, o + int(t * fs): o + int((t + step) * fs)].mean(axis=1)
                                 for t in times], axis=1)) for o in onsets])
print("Data:", X.shape, "(trials x channels x time bins)")

def generalization_matrix(X, yy, repeats=3):
    n = len(times)
    M = np.zeros((n, n))
    for seed in range(repeats):
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[:, :, 0], yy):
            for i in range(n):                                    # train time
                model = LDA(solver="lsqr", shrinkage="auto").fit(X[train, :, i], yy[train])
                for j in range(n):                                # test time
                    M[i, j] += (model.predict(X[test, :, j]) == yy[test]).sum()
    return M / (len(yy) * repeats)

def block(M, train_win, test_win):
    i = (times >= train_win[0]) & (times < train_win[1])
    j = (times >= test_win[0]) & (times < test_win[1])
    return M[np.ix_(i, j)].mean()

MAKE, RELEASE = (0.5, 1.5), (2.3, 3.0)        # seconds after cue

results = {}
for name, keep, chance in [("All 3 gestures", labels > 0, 1 / 3),
                           ("Fist vs Peace", labels != 3, 1 / 2)]:
    print(f"\n{name} (chance = {chance*100:.0f}%) ...")
    M = generalization_matrix(X[keep], labels[keep])
    results[name] = (M, chance)
    diag = np.diag(M)
    print(f"  Best moment: {diag.max()*100:.0f}% at {times[diag.argmax()]:+.1f} s")
    print(f"  Train MAKE    -> test MAKE:    {block(M, MAKE, MAKE)*100:.0f}%")
    print(f"  Train RELEASE -> test RELEASE: {block(M, RELEASE, RELEASE)*100:.0f}%")
    print(f"  Train MAKE    -> test RELEASE: {block(M, MAKE, RELEASE)*100:.0f}%   <- same code?")
    print(f"  Train RELEASE -> test MAKE:    {block(M, RELEASE, MAKE)*100:.0f}%")

# ---- Plot ----
fig, axs = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
for ax, (name, (M, chance)) in zip(axs, results.items()):
    norm = TwoSlopeNorm(vcenter=chance, vmin=0, vmax=1)      # white = chance
    im = ax.imshow(M, origin="lower", cmap="RdBu_r", norm=norm,
                   extent=[times[0], times[-1] + step, times[0], times[-1] + step])
    for v in (0, 2):                                   # cue on, cue off
        ax.axvline(v, color="k", lw=0.8, ls="--")
        ax.axhline(v, color="k", lw=0.8, ls="--")
    ax.plot([times[0], times[-1]], [times[0], times[-1]], color="k", lw=0.5)
    ax.set(xlabel="Test time (s from cue)", ylabel="Train time (s from cue)",
           title=f"{name} (chance {chance*100:.0f}%)")
    fig.colorbar(im, ax=ax, label="Decoding accuracy", shrink=0.85)
fig.suptitle("Is the gesture code stable over time?  Dashed lines = cue on (0 s) and off (2 s)")
plt.show()
