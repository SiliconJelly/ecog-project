"""
First classifier: high-gamma features + shrinkage LDA.

Features: log high-gamma power for each of the 60 channels, averaged in
250 ms time bins from 0.25 s to 2.25 s after the cue (60 x 8 = 480 features).
Classifier: LDA with automatic shrinkage (handles more features than trials).
Evaluation: 10-fold cross-validation repeated 10 times, plus a shuffled-label
control that should land at chance (33%).
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
