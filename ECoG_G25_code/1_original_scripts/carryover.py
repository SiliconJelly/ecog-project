"""
Trial-to-trial carryover: does the PREVIOUS trial shape how the NEXT
movement unfolds?

Earlier tests showed the pre-cue baseline can't tell which gesture came
before. This asks a weaker question: does the next movement's brain response
look different depending on what came before?

Part 1  Movement trials (fist/peace) after a movement vs after open hand:
        high-gamma response size, response onset, and glove reaction time
Part 2  Same check, repeat (same gesture twice) vs switch (fist <-> peace)
Part 3  Robustness: regression controlling for current gesture and trial
        number, plus a permutation test (shuffle "previous" within gesture)

Takes about 1 minute.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import stats

fs = 1200
rng = np.random.default_rng(0)
BLUE, ORANGE = "#2a78d6", "#eb6834"

# ---------------- Load, clean, high-gamma power ----------------
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog, cue, glove = y[1:61], y[61], y[62:67]
ecog = ecog - ecog.mean(axis=0)
ecog = ss.sosfiltfilt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), ecog, axis=1)
for f0 in [50, 100, 150, 200, 250, 300]:
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)
hg = ss.sosfiltfilt(ss.butter(4, [52, 300], "bandpass", fs=fs, output="sos"), ecog, axis=1) ** 2

onsets = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
labels = cue[onsets].astype(int)                 # 1 fist, 2 peace, 3 open
prev = np.r_[0, labels[:-1]]                     # 0 = no previous trial
top = [25, 36, 26, 15, 46]                       # CH26, 37, 27, 16, 47 (most active)

# High-gamma time course on the top channels, 50 ms steps, in dB
times = np.arange(-0.5, 2.5, 0.05)
step = int(0.05 * fs)
tc = 10 * np.array([[np.log10(hg[top, o + int(t * fs): o + int(t * fs) + step].mean()) for t in times]
                    for o in onsets])
size = tc[:, (times > 0.4) & (times < 1.2)].mean(axis=1)          # response size, 0.4-1.2 s
change = tc - tc[:, times < 0].mean(axis=1, keepdims=True)          # change from own pre-cue

def response_onset(row):                         # first time the rise passes half its peak
    peak = row[(times > 0) & (times < 1.5)].max()
    return times[np.argmax((times > 0) & (row > 0.5 * peak))]
onset = np.array([response_onset(r) for r in change])

g = glove.sum(axis=0)                            # all five fingers together
def reaction_time(o):
    base = g[o - int(0.5 * fs): o].mean()
    seg = np.abs(g[o: o + 2 * fs] - base)
    return np.argmax(seg > 0.3 * seg.max()) / fs
rt = np.array([reaction_time(o) for o in onsets])

has_prev = np.arange(len(labels)) > 0
movement = has_prev & (labels != 3)

def compare(title, a, b, a_name, b_name):
    print(f"\n{title}  ({a_name} n={a.sum()}, {b_name} n={b.sum()})")
    for name, x, unit in [("Response size", size, "dB"), ("Response onset", onset, "s"),
                          ("Glove reaction time", rt, "s")]:
        p = stats.mannwhitneyu(x[a], x[b]).pvalue
        print(f"  {name:20s} {x[a].mean():7.3f} vs {x[b].mean():7.3f} {unit}   diff {x[a].mean()-x[b].mean():+.3f}   p={p:.3f}")

# ---------------- Part 1 + 2 ----------------
after_move, after_open = movement & (prev != 3), movement & (prev == 3)
compare("Part 1: movement trials, after a movement vs after open hand", after_move, after_open,
        "after movement", "after open")
repeat, switch = after_move & (prev == labels), after_move & (prev != labels)
compare("Part 2: movement after movement, repeat vs switch", repeat, switch, "repeat", "switch")

# ---------------- Part 3: robustness ----------------
m = movement
X = np.c_[np.ones(m.sum()), prev[m] != 3, labels[m] == 2, np.arange(len(labels))[m] / len(labels)]
coef = np.linalg.lstsq(X, size[m], rcond=None)[0]
resid = size[m] - X @ coef
se = np.sqrt(np.diag(resid @ resid / (m.sum() - X.shape[1]) * np.linalg.inv(X.T @ X)))
p_reg = 2 * stats.t.sf(abs(coef / se), m.sum() - X.shape[1])

observed = size[after_move].mean() - size[after_open].mean()
null = []
for _ in range(5000):
    shuffled = prev.copy()
    for gesture in (1, 2):                       # shuffle "previous" within each current gesture
        k = np.where(m & (labels == gesture))[0]
        shuffled[k] = rng.permutation(shuffled[k])
    null.append(size[m & (shuffled != 3)].mean() - size[m & (shuffled == 3)].mean())
p_perm = 2 * (np.sum(np.array(null) <= observed) + 1) / 5001

print("\nPart 3: is the response-size effect robust?")
for gesture, name in [(1, "fist"), (2, "peace")]:
    d = size[after_move & (labels == gesture)].mean() - size[after_open & (labels == gesture)].mean()
    print(f"  Current {name:5s}: after movement minus after open = {d:+.2f} dB")
print(f"  Regression (controls current gesture + trial number): {coef[1]:+.2f} dB, p={p_reg[1]:.3f}")
print(f"  Trial-number effect (drift over session):             {coef[3]:+.2f} dB, p={p_reg[3]:.3f}")
print(f"  Permutation test:                                     p={p_perm:.3f}")

# ---------------- Figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})

for mask, col, name in [(after_move, BLUE, "After a movement"), (after_open, ORANGE, "After open hand")]:
    mean = change[mask].mean(axis=0)
    sem = change[mask].std(axis=0) / np.sqrt(mask.sum())
    ax[0].plot(times + 0.025, mean, color=col, lw=2, label=f"{name} (n={mask.sum()})")
    ax[0].fill_between(times + 0.025, mean - sem, mean + sem, color=col, alpha=0.2, lw=0)
ax[0].axvspan(0, 2, color="grey", alpha=0.1, label="Cue on screen")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set(xlabel="Time from cue (s)", ylabel="High-gamma change, top 5 channels (dB)",
          title="Fist/peace responses: smaller when the last trial was also a movement")
ax[0].legend(frameon=False, loc="upper right")

for x0, gesture, name in [(0, 1, "Fist"), (1, 2, "Peace")]:
    for dx, mask, col in [(-0.18, after_move, BLUE), (0.18, after_open, ORANGE)]:
        vals = size[mask & (labels == gesture)]
        ax[1].scatter(np.full(len(vals), x0 + dx) + rng.uniform(-0.06, 0.06, len(vals)), vals,
                      s=20, color=col, alpha=0.75, zorder=3)
        ax[1].plot([x0 + dx - 0.1, x0 + dx + 0.1], [vals.mean()] * 2, color="k", lw=2, zorder=4)
ax[1].set_xticks([0, 1], ["Current: fist", "Current: peace"])
ax[1].set(ylabel="Response size, 0.4-1.2 s (dB)",
          title=f"Same direction for both gestures\n(regression {coef[1]:+.2f} dB, p={p_reg[1]:.3f})")

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.show()
