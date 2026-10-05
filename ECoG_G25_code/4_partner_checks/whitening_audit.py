"""
Whitening audit (Oct 5): check the team deck's whitening numbers and what is behind them.

1. Reproduce the deck exactly (10 x 10-fold RepeatedStratifiedKFold, seed 0): 93.3 -> 96.9%.
   Same pipeline under the other CV scheme used in our report (20 x StratifiedKFold seeds 0-19).
2. Spectrum check for channel 26: does AR(10) whitening flatten the 50-300 Hz band?
   (spectral flatness = geometric / arithmetic mean of the PSD; 1 = perfectly flat)
3. Robustness: AR order 5 / 10 / 20; whitening filter fitted on the first half only;
   causal vs zero-phase downstream filters.
4. Mechanism: is the gain from whitening specifically, or from simply emphasising the
   higher frequencies inside 50-300 Hz? Compare with a fixed first-difference filter
   (x[t] - x[t-1]), which tilts the spectrum upward with no fitting at all.
"""
import json
import numpy as np
import scipy.linalg as sla
import scipy.signal as ss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
import common as C

fs = C.fs
y, cue, glove, on, off, labels = C.load_raw()
lda = lambda: LDA(solver="lsqr", shrinkage="auto")
half = int(on[44] + 4 * fs)
out = {}


def yw(x, order):
    x = x - x.mean()
    r = np.array([np.dot(x[k:], x[:len(x) - k]) / len(x) for k in range(order + 1)])
    return sla.solve_toeplitz((r[:-1], r[:-1]), r[1:])


def whiten(e, order=10, fit=slice(None)):
    o = np.empty_like(e)
    for ch in range(e.shape[0]):
        o[ch] = ss.lfilter(np.r_[1.0, -yw(e[ch, fit], order)], [1.0], e[ch])
    return o


def feats(e, causal):
    sos = ss.butter(4, [50, 300], "bandpass", fs=fs, output="sos")
    x = ss.sosfilt(sos, e, axis=1) if causal else ss.sosfiltfilt(sos, e, axis=1)
    return C.decoder_features((x ** 2).astype(np.float32), on)


def deck_cv(X):
    a = cross_val_score(lda(), X, labels, cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0))
    return float(a.mean()), float(a.std())


def report_cv(X):
    return float(np.mean([cross_val_score(lda(), X, labels, cv=StratifiedKFold(10, shuffle=True, random_state=s)).mean() for s in range(20)]))


def flat(x):
    f, p = ss.welch(x, fs=fs, nperseg=4 * fs)
    m = (f >= 55) & (f <= 295) & np.all([np.abs(f - h) > 3 for h in (100, 150, 200, 250)], axis=0)
    return float(np.exp(np.mean(np.log(p[m]))) / np.mean(p[m])), f, p


e_zp = C.clean(False)
e_c = C.clean(True)
variants = {}
for tag, e, causal in [("zero-phase", e_zp, False), ("causal", e_c, True)]:
    variants[f"{tag} | no whitening"] = feats(e, causal)
    variants[f"{tag} | AR(10) whole recording"] = feats(whiten(e, 10), causal)
    variants[f"{tag} | AR(5)"] = feats(whiten(e, 5), causal)
    variants[f"{tag} | AR(20)"] = feats(whiten(e, 20), causal)
    variants[f"{tag} | AR(10) fitted on first half only"] = feats(whiten(e, 10, slice(0, half)), causal)
    variants[f"{tag} | first-difference (no fitting)"] = feats(np.c_[e[:, :1], np.diff(e, axis=1)], causal)
print(f"{'variant':52s} {'deck CV (10x10)':>18s} {'report CV (20 seeds)':>22s}")
for k, X in variants.items():
    m, s = deck_cv(X); r = report_cv(X)
    out[k] = dict(deck_cv=m, deck_sd=s, report_cv=r)
    print(f"{k:52s} {m*100:8.1f}% ± {s*100:4.1f}   {r*100:12.1f}%", flush=True)

ch = 25
w = whiten(e_zp, 10)
fb, f, pb = flat(e_zp[ch]); fa, _, pa = flat(w[ch]); fd, _, pd = flat(np.diff(e_zp[ch]))
out["spectral_flatness_ch26_55_295Hz"] = dict(before=fb, after_AR10=fa, first_difference=fd)
lo, hi = (f >= 55) & (f <= 70), (f >= 230) & (f <= 245)
tilt = lambda p: float(10 * np.log10(p[lo].mean() / p[hi].mean()))
out["tilt_db_60_vs_240Hz_ch26"] = dict(before=tilt(pb), after_AR10=tilt(pa), first_difference=tilt(pd))
print(f"\nChannel 26, 55-295 Hz: spectral flatness before {fb:.3f}, after AR(10) {fa:.3f}, first-difference {fd:.3f} (1 = flat)")
print(f"  power at ~60 Hz minus ~240 Hz: before {tilt(pb):+.1f} dB, after AR(10) {tilt(pa):+.1f} dB, first-difference {tilt(pd):+.1f} dB")
json.dump(out, open("results/whitening_audit.json", "w"), indent=2)
