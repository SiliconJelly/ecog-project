"""
Transition relationship (Oct 5): HOW does the brain signal change from the transition into
the movement, and does that change look different for fist, peace and open?
Prosthetic angle: features of the change (speed, timing, spatial reorganisation) that track
how the hand actually moves could be continuous control signals, separate from 'which gesture'.

Per trial (causal high-gamma, cue-aligned, 50 ms bins, top-5 motor channels unless noted):
  onset_s        first time the motor high-gamma rise passes 50% of its own peak (0-1.5 s)
  slope_db_s     steepest rise (dB per second) between 0 and 1.0 s, 150 ms smoothing
  peak_db        peak change from pre-cue (dB) between 0 and 1.5 s
  t_peak_s       time of that peak
  stability_r    correlation of the 60-channel spatial pattern in the TRANSITION (0.1-0.5 s)
                 with the pattern in the MOVEMENT (0.6-1.6 s), each as change from pre-cue
                 = does the brain keep the same layout, or reorganise when the hand moves?
  lead_ms        glove movement onset minus neural onset (fist/peace only) = how far the brain is ahead
Hand (glove) per trial: movement onset, peak finger speed, movement amplitude.

Tests
  1. Do these differ between gestures? Kruskal-Wallis + fist-vs-peace permutation test
     (10,000 label shuffles within fist/peace trials).
  2. Within each gesture (identity held fixed): does the brain's rise speed / size / timing track
     how fast, how far and when the hand moves? Spearman, with session trial-number partialled out.
"""
import json
import numpy as np
import scipy.signal as ss
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

fs, step = C.fs, 0.05
rng = np.random.default_rng(61)
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(52, 300))
TOP = [25, 36, 26, 15, 46]
T = np.round(np.arange(-0.5, 2.0, step), 3)
NAMES = {1: "Fist", 2: "Peace", 3: "Open"}
COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

# motor time course (dB change from own pre-cue), all-channel windows
tc = np.array([[10 * np.log10(hg[TOP, o + int(t * fs): o + int((t + step) * fs)].mean()) for t in T] for o in on])
tc -= tc[:, T < 0].mean(1, keepdims=True)
sm = np.array([np.convolve(r, np.ones(3) / 3, mode="same") for r in tc])          # 150 ms smoothing
pre = np.array([C.win_log(hg, o, -0.5, 0) for o in on])
trans = np.array([C.win_log(hg, o, 0.1, 0.5) for o in on]) - pre
move = np.array([C.win_log(hg, o, 0.6, 1.6) for o in on]) - pre

feat = {k: np.full(90, np.nan) for k in ["onset_s", "slope_db_s", "peak_db", "t_peak_s", "stability_r"]}
w = (T >= 0) & (T <= 1.5)
for i in range(90):
    r = sm[i]
    pk = r[w].max(); ip = np.argmax(np.where(w, r, -np.inf))
    feat["peak_db"][i] = pk; feat["t_peak_s"][i] = T[ip] + step / 2
    if pk > 0:
        hit = np.where(w & (r >= 0.5 * pk))[0]
        feat["onset_s"][i] = T[hit[0]] + step / 2
    d = np.diff(r) / step
    feat["slope_db_s"][i] = d[(T[:-1] >= 0) & (T[:-1] <= 1.0)].max()
    feat["stability_r"][i] = stats.pearsonr(trans[i], move[i]).statistic

# glove behaviour
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)
g_on, g_speed, g_amp = np.full(90, np.nan), np.zeros(90), np.zeros(90)
for i, o in enumerate(on):
    base = gs[:, o - fs:o].mean(1)
    dd = np.sqrt(((gs[:, o:o + 2 * fs] - base[:, None]) ** 2).mean(0))
    g_amp[i] = np.percentile(dd, 95)
    h = np.where(dd > max(.02, .3 * g_amp[i]))[0]
    g_on[i] = h[0] / fs if len(h) else np.nan
    g_speed[i] = np.percentile(np.sqrt((np.diff(gs[:, o:o + 2 * fs], axis=1) ** 2).sum(0)) * fs, 95)
lead = (g_on - feat["onset_s"]) * 1000
lead[labels == 3] = np.nan
feat["lead_ms"] = lead

out = {"by_gesture": {}, "tests": {}, "within_gesture": {}}
print("Per-gesture medians (IQR)")
for k, v in feat.items():
    row = []
    for g in (1, 2, 3):
        x = v[(labels == g) & np.isfinite(v)]
        row.append(f"{NAMES[g]} {np.median(x):6.2f} ({np.percentile(x,25):.2f}-{np.percentile(x,75):.2f})" if len(x) else f"{NAMES[g]}   n/a")
        out["by_gesture"].setdefault(k, {})[NAMES[g]] = None if not len(x) else [float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))]
    groups = [v[(labels == g) & np.isfinite(v)] for g in (1, 2, 3) if np.sum((labels == g) & np.isfinite(v)) > 2]
    kw = stats.kruskal(*groups).pvalue if len(groups) > 1 else np.nan
    a, b = v[(labels == 1) & np.isfinite(v)], v[(labels == 2) & np.isfinite(v)]
    obs = np.median(a) - np.median(b)
    pool = np.r_[a, b]
    null = []
    for _ in range(10000):
        pp = rng.permutation(pool)
        null.append(np.median(pp[:len(a)]) - np.median(pp[len(a):]))
    p_fp = float((np.sum(np.abs(null) >= abs(obs)) + 1) / 10001)
    out["tests"][k] = dict(kruskal_p=float(kw), fist_minus_peace_median=float(obs), fist_vs_peace_p=p_fp)
    print(f"  {k:12s} " + " | ".join(row) + f"   KW p={kw:.4f}  fist-peace diff {obs:+.2f} (perm p={p_fp:.4f})")

# within-gesture coupling, partialling out trial number
def partial_spearman(x, y_, z):
    ok = np.isfinite(x) & np.isfinite(y_)
    rx, ry, rz = stats.rankdata(x[ok]), stats.rankdata(y_[ok]), stats.rankdata(z[ok])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = stats.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue), int(ok.sum())

pairs = [("slope_db_s", "hand peak speed", g_speed), ("peak_db", "hand amplitude", g_amp),
         ("onset_s", "hand onset", g_on), ("stability_r", "hand peak speed", g_speed)]
print("\nWithin-gesture coupling (Spearman, trial number partialled out)")
for g in (1, 2):
    for fk, hk, hv in pairs:
        r, p, n = partial_spearman(feat[fk][labels == g], hv[labels == g], np.arange(90)[labels == g].astype(float))
        out["within_gesture"][f"{NAMES[g]}: {fk} vs {hk}"] = dict(r=r, p=p, n=n)
        print(f"  {NAMES[g]:5s} brain {fk:12s} vs {hk:16s} r={r:+.2f}  p={p:.3f}  (n={n})")
json.dump(out, open("results/transition_relationship.json", "w"), indent=2)
np.savez("results/transition_relationship_trials.npz", labels=labels, g_on=g_on, g_speed=g_speed, g_amp=g_amp, **feat)

# ---------------- Figure ----------------
fig, ax = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
a = ax[0, 0]
for g in (1, 2, 3):
    m = labels == g
    mu, se = tc[m].mean(0), tc[m].std(0) / np.sqrt(m.sum())
    a.plot(T + step / 2, mu, color=COL[g], lw=2, label=NAMES[g])
    a.fill_between(T + step / 2, mu - se, mu + se, color=COL[g], alpha=.2, lw=0)
a.axvspan(0.1, 0.5, color="#eda100", alpha=.12); a.axvspan(0.6, 1.6, color="#1baf7a", alpha=.08)
a.text(0.3, a.get_ylim()[1] * .92, "transition", ha="center", fontsize=8); a.text(1.1, a.get_ylim()[1] * .92, "movement", ha="center", fontsize=8)
a.axvline(0, color="k", lw=.7, ls="--"); a.axhline(0, color="k", lw=.5)
a.set(xlabel="Time from cue (s)", ylabel="Motor high-gamma change (dB)", title="A. How the signal builds, by gesture")
a.legend(frameon=False)
for a, k, lab in [(ax[0, 1], "slope_db_s", "Steepest rise (dB/s)"), (ax[0, 2], "onset_s", "Neural onset (s after cue)"),
                  (ax[1, 0], "stability_r", "Transition-to-movement\nspatial pattern r")]:
    for j, g in enumerate((1, 2, 3)):
        x = feat[k][(labels == g) & np.isfinite(feat[k])]
        a.boxplot([x], positions=[j], widths=.5, showfliers=False, medianprops=dict(color="k"))
        a.scatter(np.full(len(x), j) + rng.uniform(-.12, .12, len(x)), x, s=14, color=COL[g], alpha=.7)
    t_ = out["tests"][k]
    a.set_xticks(range(3), [NAMES[g] for g in (1, 2, 3)])
    a.set(ylabel=lab, title=f"{lab.splitlines()[0]}\nfist vs peace p={t_['fist_vs_peace_p']:.3f}")
a = ax[1, 1]
for g in (1, 2):
    m = labels == g
    a.scatter(feat["slope_db_s"][m], g_speed[m], color=COL[g], s=22, label=NAMES[g])
a.set(xlabel="Brain: steepest high-gamma rise (dB/s)", ylabel="Hand: peak finger speed",
      title="B. Does a faster brain rise mean a faster hand?\n" + ", ".join(
          f"{NAMES[g]} r={out['within_gesture'][f'{NAMES[g]}: slope_db_s vs hand peak speed']['r']:+.2f}" for g in (1, 2)))
a.legend(frameon=False)
a = ax[1, 2]
for g in (1, 2):
    m = (labels == g) & np.isfinite(lead)
    a.hist(lead[m], bins=12, alpha=.55, color=COL[g], label=f"{NAMES[g]} median {np.median(lead[m]):.0f} ms")
a.axvline(0, color="k", lw=.7)
a.set(xlabel="Hand onset minus brain onset (ms)", ylabel="Trials", title=f"C. How far the brain leads the hand\nfist vs peace p={out['tests']['lead_ms']['fist_vs_peace_p']:.3f}")
a.legend(frameon=False)
for a in ax.ravel():
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/E3_transition_relationship.png", dpi=160)
