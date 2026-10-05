"""
Baseline-state analysis: "Same movement, different past"

Question: does the brain's state before a cue carry a trace of the PREVIOUS
trial, and does that state change how the CURRENT gesture is decoded?

Part 1  Decay curve: can we decode the previous gesture from the ECoG in the
        rest period after it, and for how long? (glove = posture control,
        shuffled labels = chance control)
Part 2  Beta rebound: motor-channel beta power after a real movement
        (fist/peace) vs after open hand, over the rest period.
Part 3  Baseline state: from the last 1 s before the cue, can we tell whether
        the previous trial was a movement? Does it change current decoding?

Runs in about 3-4 minutes (the permutation tests are the slow part).
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
N_PERM = 50                   # shuffled-label repeats per test (more = slower, smoother)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"

# ---------------- Load + clean (same steps as before) ----------------
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
offsets = np.where((np.diff(cue) != 0) & (cue[1:] == 0))[0] + 1
labels = cue[onsets].astype(int)                     # 1 fist, 2 peace, 3 open
motor = [15, 25, 26, 36, 46]                         # CH16, 26, 27, 37, 47 (most active)

def band_power(band, start, t0, t1, chans=slice(None)):
    """log10 mean power from start+t0 to start+t1 seconds."""
    return np.log10(power[band][chans, start + int(t0 * fs): start + int(t1 * fs)].mean(axis=1))

def all_bands(start, t0, t1):
    return np.concatenate([band_power(b, start, t0, t1) for b in bands])   # 5 bands x 60 ch

def cv_acc(F, yy, seed=0):
    model = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    return cross_val_score(model, F, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()

def perm_test(F, yy, n=N_PERM):
    real = np.mean([cv_acc(F, yy, s) for s in range(3)])
    rng = np.random.default_rng(0)
    null = np.array([cv_acc(F, rng.permutation(yy)) for _ in range(n)])
    return real, null, (np.sum(null >= real) + 1) / (n + 1)

# ---------------- Part 1: decay curve ----------------
prev_off = offsets[:-1]          # end of trial k
prev_lab = labels[:-1]           # gesture of trial k
next_on = onsets[1:]             # start of trial k+1
starts = np.arange(0, 2.01, 0.25)
res = []
print("\nPart 1: decoding the PREVIOUS gesture during the rest period (chance = 33%)")
for t0 in starts:
    t1 = t0 + 0.25
    ok = prev_off + int(t1 * fs) <= next_on          # window must end before the next cue
    F = np.array([all_bands(o, t0, t1) for o in prev_off[ok]])
    G = np.array([glove[:, o + int(t0 * fs): o + int(t1 * fs)].mean(axis=1) for o in prev_off[ok]])
    acc_e, null, p = perm_test(F, prev_lab[ok])
    acc_g = np.mean([cv_acc(G, prev_lab[ok], s) for s in range(3)])
    res.append((acc_e, np.percentile(null, 95), acc_g, p))
    print(f"  {t0:.2f}-{t1:.2f} s after cue off: ECoG {acc_e*100:4.0f}%  (shuffled 95th pct {np.percentile(null,95)*100:.0f}%, p={p:.3f})   glove {acc_g*100:4.0f}%")
res = np.array(res)
mid = starts + 0.125

# ---------------- Part 2: beta rebound ----------------
moved_before = prev_lab != 3
step = int(0.05 * fs)
tt = np.arange(0, 2.0, 0.05)
beta_tc = np.array([[band_power("beta", o, t, t + 0.05, motor).mean() for t in tt] for o in prev_off])
beta_tc = 10 * beta_tc                                 # dB
beta_tc -= beta_tc[:, -10:].mean()                     # reference: overall late-rest level

# ---------------- Part 3: baseline state -> current trial ----------------
B = np.array([all_bands(o, -1, 0) for o in onsets[1:]])      # last 1 s before each cue
moved = moved_before.astype(int)
acc_s, null_s, p_s = perm_test(B, moved, n=2 * N_PERM)
beta_base = 10 * np.array([band_power("beta", o, -1, 0, motor).mean() for o in onsets[1:]])
diff = beta_base[moved == 1].mean() - beta_base[moved == 0].mean()
u = stats.mannwhitneyu(beta_base[moved == 1], beta_base[moved == 0])

print("\nPart 3: baseline state (last 1 s before cue)")
print(f"  Motor beta after a movement vs after open hand: {diff:+.2f} dB  (Mann-Whitney p={u.pvalue:.4f})")
print(f"  Decode 'previous trial was a movement' from baseline: {acc_s*100:.0f}%"
      f"  (shuffled mean {null_s.mean()*100:.0f}%, 95th pct {np.percentile(null_s,95)*100:.0f}%, p={p_s:.3f})")

# Current-gesture decoding (same features as classify.py), split by what came before
hg = power["highgamma"]
edges_s = np.arange(0.25, 2.26, 0.25)
X = np.array([np.concatenate([np.log10(hg[:, o + int(a * fs): o + int(b * fs)].mean(axis=1))
                              for a, b in zip(edges_s[:-1], edges_s[1:])]) for o in onsets])
lda = LDA(solver="lsqr", shrinkage="auto")
correct = np.mean([cross_val_predict(lda, X, labels, cv=StratifiedKFold(10, shuffle=True, random_state=s)) == labels
                   for s in range(20)], axis=0)[1:]
print(f"  Current-gesture accuracy after a movement: {correct[moved == 1].mean()*100:.1f}%  (n={moved.sum()})")
print(f"  Current-gesture accuracy after open hand:  {correct[moved == 0].mean()*100:.1f}%  (n={(moved == 0).sum()})")

# ---------------- Figures ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

ax[0].plot(mid, res[:, 0] * 100, "-o", color=BLUE, lw=2, ms=6, label="ECoG")
ax[0].plot(mid, res[:, 2] * 100, "-o", color=ORANGE, lw=2, ms=6, label="Glove (posture)")
ax[0].plot(mid, res[:, 1] * 100, "--", color=GREY, lw=1.5, label="Shuffled labels, 95th pct")
ax[0].axhline(33.3, color=GREY, lw=0.8)
ax[0].set(xlabel="Time after previous cue ended (s)", ylabel="Accuracy (%)",
          title="Can we still decode the previous gesture?", ylim=(20, 105))
ax[0].legend(frameon=False)

ax[1].plot(tt + 0.025, beta_tc[moved_before].mean(axis=0), color=BLUE, lw=2, label="After fist / peace")
ax[1].plot(tt + 0.025, beta_tc[~moved_before].mean(axis=0), color=ORANGE, lw=2, label="After open hand")
ax[1].axhline(0, color=GREY, lw=0.8)
ax[1].set(xlabel="Time after previous cue ended (s)", ylabel="Motor beta power (dB, relative)",
          title="Beta drops during release, then rebounds")
ax[1].legend(frameon=False)

groups = [beta_base[moved == 1], beta_base[moved == 0]]
ax[2].boxplot(groups, widths=0.5, showfliers=False, medianprops=dict(color="black"))
for i, (g, c) in enumerate(zip(groups, [BLUE, ORANGE])):
    ax[2].scatter(np.full(len(g), i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, len(g)),
                  g, s=22, color=c, alpha=0.8, zorder=3)
ax[2].set_xticks([1, 2], ["After fist / peace", "After open hand"])
ax[2].set(ylabel="Motor beta, last 1 s before cue (dB)",
          title=f"Pre-cue state differs: {diff:+.1f} dB, p={u.pvalue:.3f}")

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
