"""
Checks on the partner's two strongest claims (Oct 5).

A. Spectral whitening raises accuracy 93.3% -> 96.9% (his classify_whitened.py).
   1. Paired comparison on identical CV folds, per trial (sign-flip test).
   2. Causal pipeline (forward-only filters) with and without whitening.
   3. Whitening coefficients fitted ONLY on the first half of the recording (no test-trial
      data used to build the filter), then decoding the second half chronologically.
   4. Chronological: retrain every 10 trials on all earlier trials, with vs without whitening.

B. Early ECoG (0.15-0.40 s) predicts later finger movement (his predictive_bci_insight.py).
   1. Within-gesture: is the prediction more than knowing which gesture it is?
   2. Causal filtering (no backward-pass leakage from the movement into the early window).
   3. Trials whose glove movement starts before 0.45 s removed.
"""
import json
import numpy as np
import scipy.linalg as sla
import scipy.signal as ss
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import common as C

fs = C.fs
rng = np.random.default_rng(41)
y, cue, glove, on, off, labels = C.load_raw()
out = {}


def yw(x, order=10):
    x = x - x.mean()
    r = np.array([np.dot(x[k:], x[:len(x) - k]) / len(x) for k in range(order + 1)])
    return sla.solve_toeplitz((r[:-1], r[:-1]), r[1:])


def whiten(e, fit_slice=slice(None)):
    o = np.empty_like(e)
    for ch in range(e.shape[0]):
        o[ch] = ss.lfilter(np.r_[1.0, -yw(e[ch, fit_slice])], [1.0], e[ch])
    return o


def hg_from(e, causal):
    sos = ss.butter(4, [50, 300], "bandpass", fs=fs, output="sos")
    x = ss.sosfilt(sos, e, axis=1) if causal else ss.sosfiltfilt(sos, e, axis=1)
    return (x ** 2).astype(np.float32)


lda = lambda: LDA(solver="lsqr", shrinkage="auto")


def per_trial(X, runs=20):
    c = np.zeros(90)
    for s in range(runs):
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(X, labels):
            c[te] += (lda().fit(X[tr], labels[tr]).predict(X[te]) == labels[te]) / runs
    return c


def chrono(X):
    c = []
    for s0 in range(30, 90, 10):
        tr, te = np.arange(s0), np.arange(s0, min(s0 + 10, 90))
        c.extend(lda().fit(X[tr], labels[tr]).predict(X[te]) == labels[te])
    return np.array(c, float)


def signflip(d, n=20000):
    null = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(n)])
    return float((np.sum(null >= d.mean()) + 1) / (n + 1))


half = int(on[44] + 4 * fs)          # end of trial 45
for causal in (False, True):
    tag = "causal" if causal else "zero-phase"
    e = C.clean(causal)
    X0 = C.decoder_features(hg_from(e, causal), on)
    Xw = C.decoder_features(hg_from(whiten(e), causal), on)
    Xh = C.decoder_features(hg_from(whiten(e, slice(0, half)), causal), on)
    c0, cw = per_trial(X0), per_trial(Xw)
    d = cw - c0
    r0, rw, rh = chrono(X0), chrono(Xw), chrono(Xh)
    b = int(np.sum((r0 == 0) & (rw == 1))); cc = int(np.sum((r0 == 1) & (rw == 0)))
    # second half only, filter fitted on first half
    m0 = lda().fit(X0[:45], labels[:45]); mh = lda().fit(Xh[:45], labels[:45])
    a0 = float(np.mean(m0.predict(X0[45:]) == labels[45:])); ah = float(np.mean(mh.predict(Xh[45:]) == labels[45:]))
    out[f"whitening_{tag}"] = dict(cv_original=float(c0.mean()), cv_whitened=float(cw.mean()), trials_helped=int(np.sum(d > 0)),
                                   trials_hurt=int(np.sum(d < 0)), p_signflip=signflip(d),
                                   chrono_original=float(r0.mean()), chrono_whitened=float(rw.mean()), chrono_fixed=b, chrono_broken=cc,
                                   chrono_whiten_fit_first_half=float(rh.mean()),
                                   half_split_original=a0, half_split_whiten_fit_first_half=ah)
    o = out[f"whitening_{tag}"]
    print(f"[A {tag}] CV {o['cv_original']*100:.1f}% -> {o['cv_whitened']*100:.1f}% whitened; trials helped {o['trials_helped']}, hurt {o['trials_hurt']}, p={o['p_signflip']:.4f}")
    print(f"   chronological {o['chrono_original']*100:.1f}% -> {o['chrono_whitened']*100:.1f}% ({b} fixed, {cc} broken); "
          f"filter fitted on first half only: {o['chrono_whiten_fit_first_half']*100:.1f}%")
    print(f"   train first 45 -> test last 45: {a0*100:.1f}% -> {ah*100:.1f}% (whitening filter fitted on first half only)", flush=True)

# ---- B. predictive ----
gs = ss.sosfiltfilt(ss.butter(2, 10, fs=fs, output="sos"), glove, axis=1)
gon = np.full(90, np.nan)
for i, o in enumerate(on):
    base = gs[:, o - fs:o].mean(1)
    dd = np.sqrt(((gs[:, o:o + 2 * fs] - base[:, None]) ** 2).mean(0))
    h = np.where(dd > max(.02, .3 * np.percentile(dd, 95)))[0]
    gon[i] = h[0] / fs if len(h) else np.nan
Y = np.array([glove[:, o + int(.5 * fs):o + int(2 * fs)].mean(1) - glove[:, o - int(.5 * fs):o].mean(1) for o in on])
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
for causal in (False, True):
    tag = "causal" if causal else "zero-phase"
    hg = hg_from(C.clean(causal), causal)
    Xe = np.array([np.log10(hg[:, o + int(.15 * fs):o + int(.40 * fs)].mean(1) + 1e-12) for o in on])
    Yh = cross_val_predict(ridge(), Xe, Y, cv=KFold(10, shuffle=True, random_state=0))
    allr = [stats.pearsonr(Y[:, i], Yh[:, i]).statistic for i in range(5)]
    # gesture-only baseline: predict each trial's movement as its gesture's mean (from training folds)
    Yg = np.zeros_like(Y)
    for tr, te in KFold(10, shuffle=True, random_state=0).split(Y):
        for k in (1, 2, 3):
            Yg[te[labels[te] == k]] = Y[tr][labels[tr] == k].mean(0)
    gr = [stats.pearsonr(Y[:, i], Yg[:, i]).statistic for i in range(5)]
    # within gesture: residual correlation
    within = []
    for i in range(5):
        rs = []
        for k in (1, 2, 3):
            m = labels == k
            if Y[m, i].std() > 1e-6:
                rs.append(stats.pearsonr(Y[m, i], Yh[m, i]).statistic)
        within.append(float(np.mean(rs)))
    # does early ECoG classify the gesture?
    pc = cross_val_predict(lda(), Xe, labels, cv=StratifiedKFold(10, shuffle=True, random_state=0))
    late = gon >= 0.45
    pl = cross_val_predict(lda(), Xe[late], labels[late], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out[f"predictive_{tag}"] = dict(r_all=allr, r_gesture_mean_only=gr, r_within_gesture=within,
                                    early_gesture_acc=float(np.mean(pc == labels)),
                                    early_gesture_acc_late_movers=float(np.mean(pl == labels[late])), n_late_movers=int(late.sum()))
    print(f"[B {tag}] early ECoG -> later finger change, r by finger: " + " ".join(f"{v:+.2f}" for v in allr))
    print(f"   knowing only the gesture gives r: " + " ".join(f"{v:+.2f}" for v in gr))
    print(f"   within gesture (averaged over fist/peace/open): " + " ".join(f"{v:+.2f}" for v in within))
    print(f"   early window classifies the gesture: {np.mean(pc==labels)*100:.1f}%; only trials moving at/after 0.45 s "
          f"(n={late.sum()}): {np.mean(pl==labels[late])*100:.1f}%", flush=True)
json.dump(out, open("results/partner_checks.json", "w"), indent=2)
