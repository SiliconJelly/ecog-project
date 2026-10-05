"""
Two tests (Oct 5), causal filtering throughout.

PART 1  TIMING: can the brain tell a prosthetic WHEN the hand will move?
  Detector (usable online, looks only at the past): motor high-gamma (5 top channels, 50 ms
  causal moving average) must exceed its own pre-cue mean + 3 SD for 100 ms.
  - Held out (10-fold CV): predict each fist/peace trial's glove onset from its brain onset
    (linear fit learned on training trials). Compare error with the cue-only guess
    (= median glove onset of training trials). Permutation for the correlation.
  - Also: how often does the detector fire on OPEN trials (where the hand barely moves)?
    -> a built-in 'move vs stay still' gate.

PART 2  VARIANCE: does the state just before the cue explain why the same gesture's
  movement response varies from trial to trial?
  Responses (fist + peace trials, z-scored WITHIN gesture so gesture identity is removed):
    size  = motor high-gamma 0.4-1.2 s (dB)       onset = 50%-of-peak time
    slope = steepest rise (dB/s)
  Pre-cue state predictors (last 1 s before cue):
    BRAIN: motor beta, motor high-gamma         HAND: 5-finger posture, finger speed
    CONTEXT: previous trial moved (0/1), rest gap length, trial number
  Cross-validated R^2 (leave-one-out, ridge) for: brain only, hand only, context only, all.
  Null: shuffle the response across trials (within gesture), 500 times.
"""
import json
import numpy as np
import scipy.signal as ss
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
import common as C

fs = C.fs
rng = np.random.default_rng(71)
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(52, 300))
beta = C.band_power("beta", True)
TOP = [25, 36, 26, 15, 46]
out = {}

# ---------------- PART 1 ----------------
n = int(.05 * fs)
env = np.log10(ss.lfilter(np.ones(n) / n, 1, hg[TOP].mean(0)) + 1e-12)
b_on = np.full(90, np.nan)
for i, o in enumerate(on):
    base = env[o - fs:o]
    above = env[o:o + int(1.5 * fs)] > base.mean() + 3 * base.std()
    run = np.convolve(above, np.ones(int(.1 * fs)), "valid") >= int(.1 * fs)
    idx = np.where(run)[0]
    if len(idx):
        b_on[i] = idx[0] / fs
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)
g_on, g_speed_pre, g_post = np.full(90, np.nan), np.zeros(90), np.zeros((90, 5))
for i, o in enumerate(on):
    base = gs[:, o - fs:o].mean(1)
    dd = np.sqrt(((gs[:, o:o + 2 * fs] - base[:, None]) ** 2).mean(0))
    h = np.where(dd > max(.02, .3 * np.percentile(dd, 95)))[0]
    g_on[i] = h[0] / fs if len(h) else np.nan
    g_post[i] = glove[:, o - fs:o].mean(1)
    g_speed_pre[i] = np.sqrt((np.diff(gs[:, o - fs:o], axis=1) ** 2).sum(0)).mean() * fs

fp = (labels != 3) & np.isfinite(b_on) & np.isfinite(g_on)
xb, yg = b_on[fp], g_on[fp]
pred, base_pred = np.zeros(len(xb)), np.zeros(len(xb))
for tr, te in KFold(10, shuffle=True, random_state=0).split(xb):
    m = LinearRegression().fit(xb[tr, None], yg[tr])
    pred[te] = m.predict(xb[te, None]); base_pred[te] = np.median(yg[tr])
mae, mae0 = np.mean(np.abs(pred - yg)) * 1000, np.mean(np.abs(base_pred - yg)) * 1000
r = stats.pearsonr(pred, yg).statistic
null = np.array([stats.pearsonr(xb, rng.permutation(yg)).statistic for _ in range(10000)])
p_r = float((np.sum(null >= stats.pearsonr(xb, yg).statistic) + 1) / 10001)
lead = (yg - xb) * 1000
fired_open = int(np.sum(np.isfinite(b_on[labels == 3])))
fired_move = int(np.sum(np.isfinite(b_on[labels != 3])))
out["timing"] = dict(n=int(fp.sum()), held_out_r=float(r), mae_ms=float(mae), mae_cue_only_ms=float(mae0), p_perm=p_r,
                     lead_median_ms=float(np.median(lead)), lead_iqr_ms=np.percentile(lead, [25, 75]).tolist(),
                     brain_before_hand_pct=float(np.mean(lead > 0) * 100),
                     detector_fired_movement=f"{fired_move}/60", detector_fired_open=f"{fired_open}/30")
print("PART 1  TIMING (fist/peace, held-out)")
print(f"  trials with both onsets: {fp.sum()}/60;  detector fired on movement trials {fired_move}/60, on OPEN trials {fired_open}/30")
print(f"  predict hand onset from brain onset: held-out r={r:.2f} (perm p={p_r:.4f}); error {mae:.0f} ms vs {mae0:.0f} ms using the cue alone")
print(f"  hand minus brain onset: median {np.median(lead):+.0f} ms (IQR {np.percentile(lead,25):+.0f} to {np.percentile(lead,75):+.0f}); "
      f"brain first in {np.mean(lead>0)*100:.0f}% of trials")

# ---------------- PART 2 ----------------
T = np.round(np.arange(-0.5, 2.0, 0.05), 3)
tc = np.array([[10 * np.log10(hg[TOP, o + int(t * fs): o + int((t + .05) * fs)].mean()) for t in T] for o in on])
tcc = tc - tc[:, T < 0].mean(1, keepdims=True)
sm = np.array([np.convolve(rr, np.ones(3) / 3, mode="same") for rr in tcc])
w = (T >= 0) & (T <= 1.5)
size = np.array([10 * C.win_log(hg, o, 0.4, 1.2, TOP).mean() for o in on])
onset50 = np.full(90, np.nan); slope = np.zeros(90)
for i in range(90):
    pk = sm[i][w].max()
    hit = np.where(w & (sm[i] >= .5 * pk))[0]
    onset50[i] = T[hit[0]] + .025 if pk > 0 and len(hit) else np.nan
    d = np.diff(sm[i]) / .05
    slope[i] = d[(T[:-1] >= 0) & (T[:-1] <= 1.0)].max()
pre_beta = np.array([10 * C.win_log(beta, o, -1, 0, C.MOTOR).mean() for o in on])
pre_hg = np.array([10 * C.win_log(hg, o, -1, 0, C.MOTOR).mean() for o in on])
prev = np.r_[0, labels[:-1]]
gap = np.r_[np.nan, (on[1:] - off[:-1]) / fs]
sets = {"brain (pre-cue beta + high-gamma)": np.c_[pre_beta, pre_hg],
        "hand (posture + finger speed)": np.c_[g_post, g_speed_pre],
        "context (previous moved, rest gap, trial #)": np.c_[(prev != 3).astype(float), gap, np.arange(90)],
        "all together": None}
sets["all together"] = np.c_[sets["brain (pre-cue beta + high-gamma)"], sets["hand (posture + finger speed)"], sets["context (previous moved, rest gap, trial #)"]]
keep = (labels != 3) & (np.arange(90) > 0)


def zwithin(v):
    z = np.full(90, np.nan)
    for g in (1, 2):
        m = keep & (labels == g)
        z[m] = (v[m] - np.nanmean(v[m])) / np.nanstd(v[m])
    return z


def cv_r2(X, yy):
    p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=10.0)), X, yy, cv=KFold(10, shuffle=True, random_state=0))
    return 1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2)


out["variance"] = {}
print("\nPART 2  VARIANCE explained in the movement response (fist+peace, within-gesture z-scores, 10-fold CV R^2)")
for rname, rv in [("response size", size), ("response onset", onset50), ("rise speed", slope)]:
    z = zwithin(rv)
    m = keep & np.isfinite(z)
    out["variance"][rname] = {}
    for sname, X in sets.items():
        Xm, ym = X[m], z[m]
        r2 = cv_r2(Xm, ym)
        nulls = []
        for _ in range(200):
            ys = ym.copy()
            for g in (1, 2):
                gg = labels[m] == g
                ys[gg] = rng.permutation(ys[gg])
            nulls.append(cv_r2(Xm, ys))
        nulls = np.array(nulls)
        p = float((np.sum(nulls >= r2) + 1) / 201)
        out["variance"][rname][sname] = dict(cv_r2=float(r2), null95=float(np.percentile(nulls, 95)), p=p, n=int(m.sum()))
        print(f"  {rname:15s} <- {sname:44s} R2={r2*100:+5.1f}%  (shuffle 95th {np.percentile(nulls,95)*100:+.1f}%, p={p:.3f})", flush=True)
json.dump(out, open("results/timing_variance.json", "w"), indent=2)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
for g, col, nm in [(1, "#2a78d6", "Fist"), (2, "#eb6834", "Peace")]:
    mm = labels[fp] == g
    ax[0].scatter(xb[mm], yg[mm], color=col, s=24, label=nm)
lim = [min(xb.min(), yg.min()) - .02, max(xb.max(), yg.max()) + .02]
ax[0].plot(lim, lim, "k--", lw=.7)
ax[0].set(xlabel="Brain onset, online detector (s after cue)", ylabel="Hand onset, glove (s after cue)",
          title=f"A. Brain says WHEN the hand moves\nheld-out r={r:.2f}, error {mae:.0f} ms vs {mae0:.0f} ms cue-only")
ax[0].legend(frameon=False)
ax[1].hist(lead, bins=15, color="#7a5cc4", alpha=.8)
ax[1].axvline(0, color="k", lw=.8)
ax[1].set(xlabel="Hand onset minus brain onset (ms)", ylabel="Trials",
          title=f"B. Lead time: median {np.median(lead):+.0f} ms\ndetector fired on {fired_open}/30 open trials")
names = list(sets)
xs = np.arange(len(names))
wdt = .26
for j, (rname, col) in enumerate(zip(out["variance"], ["#2a78d6", "#eda100", "#1baf7a"])):
    vals = [out["variance"][rname][s]["cv_r2"] * 100 for s in names]
    ax[2].bar(xs + (j - 1) * wdt, vals, wdt - .02, color=col, label=rname)
    for x_, s in zip(xs + (j - 1) * wdt, names):
        ax[2].plot([x_ - wdt / 2, x_ + wdt / 2], [out["variance"][rname][s]["null95"] * 100] * 2, "k--", lw=.8)
ax[2].axhline(0, color="k", lw=.6)
ax[2].set_xticks(xs, [s.split(" (")[0] for s in names])
ax[2].set(ylabel="Cross-validated R² (%)", title="C. How much of the trial-to-trial variance\ndoes the pre-cue state explain? (dashed = shuffle 95th)")
ax[2].legend(frameon=False, fontsize=8)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/E4_timing_variance.png", dpi=160)
