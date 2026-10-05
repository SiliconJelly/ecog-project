"""
Decoder test D (Oct 5): decode FINGERS, not gestures.

1. Continuous finger decoding: from high-gamma (60 ch, 100 ms bins, causal, current bin
   + 3 previous bins = 240 features) predict all 5 glove fingers at every 100 ms from
   -0.5 to 3.5 s around each cue. Ridge regression, CV by TRIAL (10 folds).
   Reported: r per finger, all trials and within fist/peace trials only.
2. Gestures via fingers: predict each trial's hold-posture finger vector, then label the
   trial by the nearest gesture template (templates from training trials). Compare with
   the 3-class LDA on the same features/splits.
3. Leave-one-gesture-out: train the finger decoder on two gestures and predict the third.
   With only 3 gestures, some fingers never vary in the training set, so this is a test
   of what this dataset CAN show, reported honestly.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import common as C

fs = C.fs
y, cue, glove, on, off, labels = C.load_raw()
hg = C.band_power(None, True, lo_hi=(50, 300))
step, LAGS = 0.1, 4
T = np.arange(-0.5, 3.5, step)
FC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def binned(o, t0):
    return np.log10(hg[:, o + int(t0 * fs): o + int((t0 + step) * fs)].mean(1))


Xt = np.array([[np.concatenate([binned(o, t - l * step) for l in range(LAGS)]) for t in T] for o in on])  # trials x T x 240
Yt = np.array([[glove[:, o + int(t * fs): o + int((t + step) * fs)].mean(1) for t in T] for o in on])     # trials x T x 5
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 16)))

# 1. continuous decoding, CV by trial
pred = np.zeros_like(Yt)
for tr, te in KFold(10, shuffle=True, random_state=0).split(np.arange(90)):
    m = ridge().fit(Xt[tr].reshape(-1, Xt.shape[2]), Yt[tr].reshape(-1, 5))
    pred[te] = m.predict(Xt[te].reshape(-1, Xt.shape[2])).reshape(len(te), len(T), 5)
cont = {}
for scope, mask in [("all trials", np.ones(90, bool)), ("fist/peace only", labels != 3)]:
    cont[scope] = {f: float(stats.pearsonr(pred[mask][:, :, i].ravel(), Yt[mask][:, :, i].ravel()).statistic)
                   for i, f in enumerate(C.FINGERS)}
    print(f"Continuous finger decoding, {scope:16s}: " + "  ".join(f"{f} r={v:.2f}" for f, v in cont[scope].items()))

# 2. gestures via fingers
H = (T >= 0.75) & (T < 1.75)
Xh = Xt[:, H, :LAGS * 0 + 60].mean(1)            # current-bin features averaged over hold (60 ch)
Xh = np.concatenate([Xt[:, H, :60].mean(1), Xt[:, (T >= 0.25) & (T < 0.75), :60].mean(1)], 1)
Yh = Yt[:, H].mean(1)
acc_f, acc_l = [], []
for s in range(10):
    pf, pl = np.zeros(90, int), np.zeros(90, int)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(Xh, labels):
        m = ridge().fit(Xh[tr], Yh[tr])
        tmpl = np.array([Yh[tr][labels[tr] == k].mean(0) for k in (1, 2, 3)])
        yp = m.predict(Xh[te])
        pf[te] = 1 + np.argmin(((yp[:, None, :] - tmpl[None]) ** 2).sum(2), 1)
        pl[te] = LDA(solver="lsqr", shrinkage="auto").fit(Xh[tr], labels[tr]).predict(Xh[te])
    acc_f.append(np.mean(pf == labels)); acc_l.append(np.mean(pl == labels))
print(f"Gesture via finger decoder + nearest template: {np.mean(acc_f)*100:.1f}%   direct 3-class LDA, same features: {np.mean(acc_l)*100:.1f}%")

# 3. leave-one-gesture-out
logo = {}
names = {1: "Fist", 2: "Peace", 3: "Open"}
for held in (1, 2, 3):
    tr = labels != held
    sd = Yh[tr].std(0)
    m = ridge().fit(Xh[tr], Yh[tr])
    yp = m.predict(Xh[labels == held]).mean(0)
    actual = Yh[labels == held].mean(0)
    templates = {names[k]: Yh[labels == k].mean(0) for k in (1, 2, 3)}
    nearest = min(templates, key=lambda k: np.sum((templates[k] - yp) ** 2))
    logo[names[held]] = dict(predicted=yp.tolist(), actual=actual.tolist(), training_sd=sd.tolist(), nearest_template=nearest)
    print(f"Held out {names[held]:5s}: predicted " + " ".join(f"{v:.2f}" for v in yp) + "  | actual " +
          " ".join(f"{v:.2f}" for v in actual) + f"  | nearest template: {nearest}" +
          f"  | fingers with ~no training variance: {[f for f, s in zip(C.FINGERS, sd) if s < .08]}")
json.dump(dict(continuous_r=cont, gesture_via_fingers=float(np.mean(acc_f)), direct_lda=float(np.mean(acc_l)), leave_one_gesture_out=logo),
          open("results/finger_decoder.json", "w"), indent=2)

# figure: example trials + r bars + LOGO
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
ex = [np.where(labels == k)[0][5] for k in (1, 2, 3)]
off_ = 0
for i in ex:
    tt = T + off_
    for fi in (1, 3):
        ax[0].plot(tt, Yt[i, :, fi], color=FC[fi], lw=2, label=f"{C.FINGERS[fi]} (glove)" if i == ex[0] else None)
        ax[0].plot(tt, pred[i, :, fi], color=FC[fi], lw=1.5, ls="--", label=f"{C.FINGERS[fi]} (decoded)" if i == ex[0] else None)
    ax[0].text(off_ + 1.5, 1.08, names[labels[i]], ha="center")
    off_ += 4.5
ax[0].set(ylim=(-.1, 1.15), xticks=[], ylabel="Finger bend", title="A. Decoded vs actual index and ring, one fist, peace, open trial")
ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="center right")
xs = np.arange(5)
for j, (scope, col) in enumerate(zip(cont, ["#8a8a85", "#2a78d6"])):
    ax[1].bar(xs + (j - .5) * .38, list(cont[scope].values()), .35, color=col, label=scope)
ax[1].set_xticks(xs, C.FINGERS)
ax[1].set(ylim=(0, 1), ylabel="r (decoded vs glove, every 100 ms)", title="B. Continuous finger decoding")
ax[1].legend(frameon=False, fontsize=8)
for j, (k, col) in enumerate(zip(("Fist", "Peace", "Open"), ["#2a78d6", "#eb6834", "#1baf7a"])):
    ax[2].plot(xs + j * 6, logo[k]["actual"], "o-", color=col, lw=2)
    ax[2].plot(xs + j * 6, logo[k]["predicted"], "s--", color=col, lw=1.5, mfc="white")
    ax[2].text(j * 6 + 2, 1.08, f"held out: {k}", ha="center", fontsize=9)
ax[2].set_xticks(np.r_[xs, xs + 6, xs + 12], [f[0] for f in C.FINGERS] * 3)
ax[2].set(ylim=(-.1, 1.15), ylabel="Finger bend during hold", title="C. Never-seen gesture: actual (solid) vs predicted (dashed)")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/D5_finger_decoder.png", dpi=160)
