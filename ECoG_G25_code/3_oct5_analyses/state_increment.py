"""
Does pre-cue BRAIN state add held-out explained variance in response size beyond history,
posture, hand kinematics and recording time? Nested comparison on identical 10-fold splits
(repeated 20x), fist+peace, within-gesture z-scored response. Null: shuffle only the brain
columns across trials (within gesture), 300x -> tests the increment itself.
Also: continuous false-alarm rate of the online 'movement' detector during rest.
"""
import numpy as np, json, sys
src = open("timing_variance.py").read().split("out[\"variance\"] = {}")[0]
g = {"__name__": "tv"}; exec(compile(src, "tv", "exec"), g)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
import scipy.signal as ss
rng = np.random.default_rng(91)
labels, keep, size, pre_beta, pre_hg = g["labels"], g["keep"], g["size"], g["pre_beta"], g["pre_hg"]
z = g["zwithin"](size); m = keep & np.isfinite(z) & np.isfinite(g["gap"])
fs, on, gs = g["fs"], g["on"], g["gs"]
# hand kinematics of the CURRENT movement (amplitude, peak speed) as extra controls
amp = np.zeros(90); spd = np.zeros(90)
for i, o in enumerate(on):
    base = gs[:, o - fs:o].mean(1)
    dd = np.sqrt(((gs[:, o:o + 2 * fs] - base[:, None]) ** 2).mean(0)); amp[i] = np.percentile(dd, 95)
    spd[i] = np.percentile(np.sqrt((np.diff(gs[:, o:o + 2 * fs], axis=1) ** 2).sum(0)) * fs, 95)
prev = g["prev"]
H = np.c_[(prev != 3).astype(float), g["gap"]]                # history
P = np.c_[g["g_post"], g["g_speed_pre"]]                      # pre-cue posture
K = np.c_[amp, spd]                                           # current movement kinematics
T = np.arange(90)[:, None].astype(float)                      # recording time
B = np.c_[pre_beta, pre_hg]                                   # brain state
def r2(X, yy, seeds=range(20)):
    out = []
    for s in seeds:
        p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=10.0)), X, yy, cv=KFold(10, shuffle=True, random_state=s))
        out.append(1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2))
    return float(np.mean(out))
yy = z[m]; lab = labels[m]
base = np.c_[H, P, K, T][m]
full = np.c_[H, P, K, T, B][m]
r_base, r_full = r2(base, yy), r2(full, yy)
inc = r_full - r_base
null = []
for _ in range(300):
    Bs = B[m].copy()
    for gg in (1, 2):
        ix = np.where(lab == gg)[0]; Bs[ix] = Bs[rng.permutation(ix)]
    null.append(r2(np.c_[base, Bs], yy, seeds=range(5)) - r2(base, yy, seeds=range(5)))
null = np.array(null); p = float((np.sum(null >= inc) + 1) / 301)
r_H, r_B, r_HB = r2(H[m], yy), r2(B[m], yy), r2(np.c_[H, B][m], yy)
print(f"held-out R2 (20x10-fold): history {r_H*100:+.1f}%  brain {r_B*100:+.1f}%  history+brain {r_HB*100:+.1f}%")
print(f"controls (history+posture+kinematics+time) {r_base*100:+.1f}%  -> + brain state {r_full*100:+.1f}%  increment {inc*100:+.1f} pts (shuffle-brain 95th {np.percentile(null,95)*100:+.1f}, p={p:.3f})")
print(f"corr(pre-cue beta, previous moved) within these trials: r={np.corrcoef(pre_beta[m], H[m,0])[0,1]:+.2f}")
# false alarms during rest
env = g["env"]; off = g["off"]; nx = np.r_[on[1:], len(env)]
fa = 0; secs = 0.0
for i in range(90):
    a, b = off[i] + int(1.0 * fs), nx[i] - int(0.2 * fs)       # rest, >=1 s after cue off, before next cue
    if b - a < fs: continue
    base_ = env[on[i] - fs:on[i]]                          # the same threshold the detector used on trial i (its pre-cue rest)
    above = env[a:b] > base_.mean() + 3 * base_.std()
    run = np.convolve(above, np.ones(int(.1 * fs)), "valid") >= int(.1 * fs)
    fa += int(np.sum(np.diff(np.r_[0, run.astype(int)]) == 1)); secs += (b - a) / fs
print(f"rest false alarms: {fa} in {secs:.0f} s of rest = {fa/secs*60:.1f} per minute")
json.dump(dict(r2_history=r_H, r2_brain=r_B, r2_history_brain=r_HB, r2_controls=r_base, r2_controls_plus_brain=r_full,
               increment=inc, increment_null95=float(np.percentile(null, 95)), p=p, rest_false_alarms=fa, rest_seconds=secs),
          open("results/state_increment.json", "w"), indent=2)
