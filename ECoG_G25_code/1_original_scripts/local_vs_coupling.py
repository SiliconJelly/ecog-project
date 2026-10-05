"""
Local vs relational: where is the information stored?

LOCAL      = how strong each electrode's activity is on its own (power)
RELATIONAL = how much each electrode's activity rises and falls together with
             the others (coupling = average correlation with all other electrodes)
SCRAMBLED  = coupling after randomly time-shifting each electrode's signal.
             Each electrode keeps exactly the same activity, but the timing
             between electrodes is broken. If coupling decoding drops to chance,
             the information really was in the coordination.

Tested on three questions:
  1. Baseline (last 1 s before cue): did the previous trial involve movement?
  2. Rest right after a gesture (0.25-1.25 s): which gesture was it?
  3. During the movement (0.25-1.75 s): which gesture is it?

Uses beta (13-30 Hz) and high-gamma (52-300 Hz). Takes about 6-8 minutes.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

fs, fe = 1200, 100                      # original and downsampled rate
rng = np.random.default_rng(0)

# ---- Load + clean ----
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog, cue = y[1:61], y[61]
ecog = ecog - ecog.mean(axis=0)
ecog = ss.sosfiltfilt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), ecog, axis=1)
for f0 in [50, 100, 150, 200, 250, 300]:
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)

# ---- Amplitude envelope per band (how strong the rhythm is, moment by moment) ----
env = {}
for name, (lo, hi) in {"beta": (13, 30), "highgamma": (52, 300)}.items():
    x = ss.sosfiltfilt(ss.butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), ecog, axis=1)
    env[name] = ss.decimate(np.abs(ss.hilbert(x, axis=1)), fs // fe, axis=1)   # -> 100 Hz
print("Envelopes done.")

onsets = (np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1) // (fs // fe)
offsets = (np.where((np.diff(cue) != 0) & (cue[1:] == 0))[0] + 1) // (fs // fe)
labels = cue[np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1].astype(int)

def features(start, t0, t1, scramble=False):
    power, coupling = [], []
    for band in env:
        seg = np.log(np.maximum(env[band][:, start + int(t0 * fe): start + int(t1 * fe)], 1e-3))
        if scramble:                                   # shift each channel by a random amount
            seg = np.stack([np.roll(ch, rng.integers(seg.shape[1])) for ch in seg])
        power.append(seg.mean(axis=1))
        R = np.corrcoef(seg)
        np.fill_diagonal(R, np.nan)
        coupling.append(np.nanmean(R, axis=1))         # avg correlation with all other channels
    return np.concatenate(power), np.concatenate(coupling)

def accuracy(F, yy, repeats=3):
    model = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    return np.mean([cross_val_score(model, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
                    for s in range(repeats)])

def shuffled_95(F, yy, n=20):
    model = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    null = [cross_val_score(model, F, rng.permutation(yy), cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            for _ in range(n)]
    return np.percentile(null, 95)

prev = labels[:-1]
tests = [("1. Baseline:\nprevious trial moved?", onsets[1:], -1.0, 0.0, (prev != 3).astype(int)),
         ("2. Rest after gesture:\nwhich gesture was it?", offsets[:-1], 0.25, 1.25, prev),
         ("3. During movement:\nwhich gesture is it?", onsets, 0.25, 1.75, labels)]

rows = []
for title, starts, t0, t1, yy in tests:
    F = [features(s, t0, t1) for s in starts]
    P = np.array([f[0] for f in F])
    C = np.array([f[1] for f in F])
    Cs = np.array([features(s, t0, t1, scramble=True)[1] for s in starts])
    r = dict(power=accuracy(P, yy), coupling=accuracy(C, yy), scrambled=accuracy(Cs, yy),
             both=accuracy(np.c_[P, C], yy), null=shuffled_95(P, yy))
    rows.append(r)
    print(f"\n{title.replace(chr(10), ' ')}")
    print(f"  Local power:          {r['power']*100:.0f}%")
    print(f"  Coupling:             {r['coupling']*100:.0f}%")
    print(f"  Coupling, scrambled:  {r['scrambled']*100:.0f}%")
    print(f"  Power + coupling:     {r['both']*100:.0f}%")
    print(f"  Shuffled labels 95th: {r['null']*100:.0f}%   (beat this to count)")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 4.8))
kinds = [("power", "Local power", "#2a78d6"), ("coupling", "Coupling", "#eb6834"),
         ("scrambled", "Coupling, timing scrambled", "#c3c2b7")]
w = 0.26
for k, (key, lab, col) in enumerate(kinds):
    vals = [r[key] * 100 for r in rows]
    ax.bar(np.arange(3) + (k - 1) * w, vals, w - 0.03, color=col, label=lab)
for i, r in enumerate(rows):
    ax.plot([i - 1.5 * w, i + 1.5 * w], [r["null"] * 100] * 2, color="k", lw=1.5, ls="--",
            label="Shuffled-label 95th pct" if i == 0 else None)
ax.set_xticks(range(3), [t[0] for t in tests])
ax.set(ylabel="Decoding accuracy (%)", ylim=(0, 100),
       title="Where is the information: single electrodes or their coordination?")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="upper left")
plt.tight_layout()
plt.show()
