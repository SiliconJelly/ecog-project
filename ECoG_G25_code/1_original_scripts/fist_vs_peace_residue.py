"""
Does the leftover state tell FIST apart from PEACE?

baseline_state.py showed the brain keeps an "I just moved" trace before the
next cue (after fist/peace vs after open hand). This script asks the harder
question: is that trace different after a fist than after a peace sign?
And if so, can it reduce the decoder's fist <-> peace mix-ups?

Part 1  Pre-cue motor beta, split by previous gesture (fist / peace / open)
Part 2  Decode previous fist vs peace from the last 1 s before the cue:
        brain signal vs glove (hand posture), with a shuffled-label control
Part 3  Fist <-> peace errors for the standard decoder vs versions that use
        the baseline or the previous gesture

Takes about 2-3 minutes.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

fs = 1200
rng = np.random.default_rng(0)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"

# ---------------- Load + clean ----------------
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog, cue, glove = y[1:61], y[61], y[62:67]
ecog = ecog - ecog.mean(axis=0)
ecog = ss.sosfiltfilt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), ecog, axis=1)
for f0 in [50, 100, 150, 200, 250, 300]:
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)

bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "lowgamma": (30, 48), "highgamma": (52, 300)}
power = {}
for name, (lo, hi) in bands.items():
    x = ss.sosfiltfilt(ss.butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), ecog, axis=1)
    power[name] = (x ** 2).astype(np.float32)
print("Filtering done.")

onsets = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
labels = cue[onsets].astype(int)                  # 1 fist, 2 peace, 3 open
prev = np.r_[0, labels[:-1]]                      # previous trial's gesture (0 = none)
motor = [15, 25, 26, 36, 46]                      # CH16, 26, 27, 37, 47

def band_power(band, start, t0, t1, chans=slice(None)):
    return np.log10(power[band][chans, start + int(t0 * fs): start + int(t1 * fs)].mean(axis=1))

def cv_acc(F, yy, seed=0):
    model = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    return cross_val_score(model, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()

# ---------------- Part 1: pre-cue beta by previous gesture ----------------
beta = np.array([10 * band_power("beta", o, -1, 0, motor).mean() for o in onsets])
groups = {name: beta[prev == k] for k, name in [(1, "Fist"), (2, "Peace"), (3, "Open")]}
u = stats.mannwhitneyu(groups["Fist"], groups["Peace"])
print("\nPart 1: motor beta in the last 1 s before the cue, by PREVIOUS gesture")
for name, g in groups.items():
    print(f"  after {name:5s}: {g.mean():.2f} dB  (n={len(g)})")
print(f"  Fist vs Peace difference: {groups['Fist'].mean() - groups['Peace'].mean():+.2f} dB  (p={u.pvalue:.2f})")

# ---------------- Part 2: decode previous fist vs peace from the baseline ----------------
fp = (prev == 1) | (prev == 2)
yy = prev[fp]
B = np.array([np.concatenate([band_power(b, o, -1, 0) for b in bands]) for o in onsets[fp]])
G = np.array([glove[:, o - fs: o].mean(axis=1) for o in onsets[fp]])
acc_brain = np.mean([cv_acc(B, yy, s) for s in range(5)])
acc_glove = np.mean([cv_acc(G, yy, s) for s in range(5)])
null = np.array([cv_acc(B, rng.permutation(yy)) for _ in range(100)])
p_brain = (np.sum(null >= acc_brain) + 1) / 101
print("\nPart 2: was the previous trial a fist or a peace sign? (chance = 50%)")
print(f"  Brain (pre-cue ECoG): {acc_brain*100:.0f}%  (shuffled 95th pct {np.percentile(null,95)*100:.0f}%, p={p_brain:.2f})")
print(f"  Glove (hand posture): {acc_glove*100:.0f}%")

# ---------------- Part 3: can baseline info cut fist <-> peace errors? ----------------
edges = np.arange(0.25, 2.26, 0.25)
X = np.array([np.concatenate([band_power("highgamma", o, a, b) for a, b in zip(edges[:-1], edges[1:])])
              for o in onsets])
base_hg = np.array([band_power("highgamma", o, -1, 0) for o in onsets])
base_beta = np.array([band_power("beta", o, -1, 0) for o in onsets])
prev_onehot = np.eye(4)[prev][:, 1:]

versions = {"Standard": X,
            "+ previous gesture": np.c_[X, prev_onehot],
            "+ baseline high-gamma": np.c_[X, base_hg],
            "+ baseline beta": np.c_[X, base_beta],
            "Baseline-normalized": X - np.tile(base_hg, len(edges) - 1)}
errors = {}
print("\nPart 3: fist <-> peace mix-ups (average per full cross-validation run, 20 runs)")
for name, F in versions.items():
    fp_err, acc = 0, 0
    for s in range(20):
        pred = cross_val_predict(LDA(solver="lsqr", shrinkage="auto"), F, labels,
                                 cv=StratifiedKFold(10, shuffle=True, random_state=s))
        fp_err += np.sum(((labels == 1) & (pred == 2)) | ((labels == 2) & (pred == 1)))
        acc += np.mean(pred == labels)
    errors[name] = fp_err / 20
    print(f"  {name:22s} {fp_err/20:4.1f} errors   accuracy {acc/20*100:.1f}%")

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

g = list(groups.values())
ax[0].boxplot(g, widths=0.5, showfliers=False, medianprops=dict(color="black"))
for i, (vals, col) in enumerate(zip(g, [BLUE, ORANGE, AQUA])):
    ax[0].scatter(np.full(len(vals), i + 1) + rng.uniform(-0.12, 0.12, len(vals)), vals,
                  s=20, color=col, alpha=0.8, zorder=3)
ax[0].set_xticks([1, 2, 3], ["After fist", "After peace", "After open"])
ax[0].set(ylabel="Motor beta, last 1 s before cue (dB)",
          title=f"Fist vs peace: {groups['Fist'].mean()-groups['Peace'].mean():+.2f} dB (p={u.pvalue:.2f})")

ax[1].bar([0, 1], [acc_brain * 100, acc_glove * 100], color=[BLUE, ORANGE], width=0.6)
ax[1].axhline(np.percentile(null, 95) * 100, color="k", ls="--", lw=1.5, label="Shuffled-label 95th pct")
ax[1].axhline(50, color=GREY, lw=1, label="Chance")
for i, v in enumerate([acc_brain, acc_glove]):
    ax[1].text(i, v * 100 + 2, f"{v*100:.0f}%", ha="center")
ax[1].set_xticks([0, 1], ["Brain (ECoG)", "Hand (glove)"])
ax[1].set(ylabel="Accuracy (%)", ylim=(0, 105), title="Before the cue: was the last trial fist or peace?")
ax[1].legend(frameon=False, loc="upper left")

names = list(errors)
ax[2].barh(range(len(names))[::-1], [errors[n] for n in names],
           color=[BLUE] + [GREY] * (len(names) - 1))
for i, n in enumerate(names):
    ax[2].text(errors[n] + 0.1, len(names) - 1 - i, f"{errors[n]:.1f}", va="center")
ax[2].set_yticks(range(len(names))[::-1], names)
ax[2].set(xlabel="Fist <-> peace errors per run (of 60 trials)", title="Baseline info doesn't reduce mix-ups")

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.show()
