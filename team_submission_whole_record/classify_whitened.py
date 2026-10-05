"""
First classifier: high-gamma features + shrinkage LDA.

Features: log high-gamma power for each of the 60 channels, averaged in
250 ms time bins from 0.25 s to 2.25 s after the cue (60 x 8 = 480 features).
Classifier: LDA with automatic shrinkage (handles more features than trials).
Evaluation: 10-fold cross-validation repeated 10 times, plus a shuffled-label
control that should land at chance (33%).
"""
WHITEN = True

import numpy as np
import scipy.io as sio
import scipy.signal as ss
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def whiten_whole_recording(channel_signals):
    order = 10
    sample_count = channel_signals.shape[1]
    whitened_signals = np.empty_like(channel_signals)
    coefficients = np.empty((channel_signals.shape[0], order + 1))
    for channel_index, channel_signal in enumerate(channel_signals):
        centered = channel_signal - channel_signal.mean()
        covariance = np.asarray([
            np.dot(centered[:sample_count - lag], centered[lag:]) / sample_count
            for lag in range(order + 1)
        ])
        if not np.isfinite(covariance).all() or covariance[0] <= 0:
            raise ValueError(f"Cannot fit AR(10) to channel {channel_index + 1}.")
        covariance /= covariance[0]
        predictor = solve_toeplitz(covariance[:-1], covariance[1:])
        coefficients[channel_index] = np.r_[1.0, -predictor]
        whitened_signals[channel_index] = ss.lfilter(
            coefficients[channel_index], [1.0], channel_signal
        )
    if not np.isfinite(whitened_signals).all():
        raise ValueError("Whitening produced non-finite values.")
    return whitened_signals, coefficients


fs = 1200
y = sio.loadmat("ECoG_Handpose.mat")["y"]
ecog, cue = y[1:61], y[61]
names = ["Fist", "Peace", "Open"]

# ---- Preprocessing (same as high_gamma.py) ----
ecog = ecog - ecog.mean(axis=0)
ecog = ss.sosfiltfilt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), ecog, axis=1)
for f0 in [50, 100, 150, 200, 250, 300]:
    b, a = ss.iirnotch(f0, Q=30, fs=fs)
    ecog = ss.filtfilt(b, a, ecog, axis=1)

if WHITEN:
    spectrum_frequencies, spectrum_before = ss.welch(ecog[25], fs=fs, nperseg=2400)
    ecog, whitening_coefficients = whiten_whole_recording(ecog)
    _, spectrum_after = ss.welch(ecog[25], fs=fs, nperseg=2400)
    spectrum_figure, spectrum_axis = plt.subplots(figsize=(9, 5))
    positive_frequencies = spectrum_frequencies > 0
    spectrum_axis.loglog(spectrum_frequencies[positive_frequencies], spectrum_before[positive_frequencies], label="Before whitening")
    spectrum_axis.loglog(spectrum_frequencies[positive_frequencies], spectrum_after[positive_frequencies], label="After whitening")
    spectrum_axis.axvspan(50, 300, color="#dddddd", alpha=0.35, label="Classifier band")
    spectrum_axis.set(xlabel="Frequency (Hz)", ylabel="Power spectral density (signal units squared / Hz)",
                      title="Channel 26: AR(10) whitening fitted to the whole recording", xlim=(1, 600))
    spectrum_axis.legend(frameon=False)
    spectrum_figure.tight_layout()
    print(f"Whitening: AR(10), Yule-Walker, whole recording ({ecog.shape[1]} samples/channel), one-pass lfilter.")

hg = ss.sosfiltfilt(ss.butter(4, [50, 300], "bandpass", fs=fs, output="sos"), ecog, axis=1)
hg = hg ** 2

# ---- Features ----
onsets = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
labels = cue[onsets].astype(int)
bins = np.arange(0.25, 2.26, 0.25)                    # bin edges in seconds

def features(o):
    edges = (o + bins * fs).astype(int)
    pw = [hg[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])]
    return np.log10(np.stack(pw, axis=1)).ravel()     # 60 ch x 8 bins

X = np.array([features(o) for o in onsets])
print("Feature matrix:", X.shape, "(trials x features)")

# ---- Classify ----
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)

acc = cross_val_score(lda, X, labels, cv=cv)
print(f"\n3-class accuracy:      {acc.mean()*100:.1f}% +/- {acc.std()*100:.1f}%   (chance = 33.3%)")

fp = labels != 3
acc_fp = cross_val_score(lda, X[fp], labels[fp], cv=cv)
print(f"Fist vs Peace only:    {acc_fp.mean()*100:.1f}% +/- {acc_fp.std()*100:.1f}%   (chance = 50%)")

rng = np.random.default_rng(0)
acc_shuf = [cross_val_score(lda, X, rng.permutation(labels), cv=StratifiedKFold(10, shuffle=True, random_state=i)).mean()
            for i in range(20)]
print(f"Shuffled-label control: {np.mean(acc_shuf)*100:.1f}%   (should be near 33%)")

# ---- Confusion matrix ----
pred = cross_val_predict(lda, X, labels, cv=StratifiedKFold(10, shuffle=True, random_state=0))
cm = confusion_matrix(labels, pred)
print("\nConfusion matrix (rows = true, columns = predicted):")
print("         " + "  ".join(f"{n:>5}" for n in names))
for n, row in zip(names, cm):
    print(f"{n:>7}  " + "  ".join(f"{v:5d}" for v in row))

ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues", colorbar=False)
plt.title(f"High-gamma + LDA: {acc.mean()*100:.1f}% (10-fold CV)")
plt.tight_layout()
plt.show()
