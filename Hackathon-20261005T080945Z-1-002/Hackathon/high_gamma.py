"""
High-gamma power per gesture.

1. Clean the raw ECoG (common average reference, remove 50 Hz line noise)
2. Keep only the high-gamma band (50-300 Hz) and measure its power over time
3. Cut out each trial (-1 s to +3 s around the cue)
4. Compare each trial to its own pre-cue baseline (in dB)
5. Plot: which electrodes light up for each gesture, and when
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt

fs = 1200
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog = y[1:61]                 # 60 ECoG channels
cue = y[61]                    # 0 rest, 1 fist, 2 peace, 3 open
names = {1: "Fist (rock)", 2: "Peace (scissors)", 3: "Open (paper)"}

# ---- 1. Clean ----
ecog = ecog - ecog.mean(axis=0)                       # common average reference
sos = ss.butter(4, 1, "highpass", fs=fs, output="sos")
ecog = ss.sosfiltfilt(sos, ecog, axis=1)              # remove slow drift
for f0 in [50, 100, 150, 200, 250, 300]:              # 50 Hz mains + harmonics
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)

# ---- 2. High-gamma power ----
sos = ss.butter(4, [50, 300], "bandpass", fs=fs, output="sos")
hg = ss.sosfiltfilt(sos, ecog, axis=1)
win = int(0.05 * fs)                                  # 50 ms smoothing
hg = ss.fftconvolve(hg**2, np.ones((1, win)) / win, mode="same", axes=1)
hg = 10 * np.log10(hg)

# ---- 3. Cut trials ----
onsets = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
labels = cue[onsets].astype(int)
pre, post = int(1 * fs), int(3 * fs)
trials = np.stack([hg[:, o - pre:o + post] for o in onsets])   # trials x ch x time
times = np.arange(-pre, post) / fs
print("Trials:", trials.shape[0], " per gesture:", {names[k]: int((labels == k).sum()) for k in names})

# ---- 4. Baseline-correct: last 1 s before the cue ----
base = trials[:, :, times < 0].mean(axis=2, keepdims=True)
trials = trials - base                                 # dB change from baseline

# ---- 5a. Electrode maps: mean change 0.5-2.5 s after cue ----
active = (times >= 0.5) & (times <= 2.5)
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
for ax, k in zip(axs, names):
    m = trials[labels == k][:, :, active].mean(axis=(0, 2))    # one value per channel
    grid = m.reshape(6, 10).T          # ch1-10 = first column (top to bottom), 51-60 = last
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-3, vmax=3)
    for ch in range(60):
        ax.text(ch // 10, ch % 10, ch + 1, ha="center", va="center", fontsize=7)
    ax.set_title(names[k]); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, label="High-gamma change from baseline (dB)", shrink=0.8)
fig.suptitle("Which electrodes activate for each gesture (0.5-2.5 s after cue)")

# ---- 5b. Time course of the most active channels ----
overall = trials[:, :, active].mean(axis=(0, 2))
top = np.argsort(overall)[::-1][:5]
print("Most active channels (CH numbering 1-60):", (top + 1).tolist())
for c in top:
    print(f"  CH{c + 1:2d}: +{overall[c]:.2f} dB")

plt.figure(figsize=(9, 4))
for k in names:
    mean_tc = trials[labels == k][:, top].mean(axis=(0, 1))
    plt.plot(times, mean_tc, label=names[k])
plt.axvspan(0, 2, color="grey", alpha=0.15, label="Cue on screen")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from cue (s)"); plt.ylabel("High-gamma change (dB)")
plt.title(f"High-gamma over time, top 5 channels {(top + 1).tolist()}")
plt.legend(); plt.tight_layout()
plt.show()
