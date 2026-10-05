"""
Shared loading + preprocessing for the Oct 5 tests.

CAUSAL = True  -> every filter runs forward only (scipy sosfilt / lfilter).
                  A sample at time t is only influenced by samples at or before t,
                  so nothing that happens after the cue can leak into the pre-cue window.
CAUSAL = False -> the original zero-phase pipeline (sosfiltfilt / filtfilt),
                  kept so results can be compared one-to-one.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
from pathlib import Path

fs = 1200
DATA = Path(__file__).resolve().parent / "ECoG_Handpose.mat"
MOTOR = [15, 25, 26, 36, 46]           # CH16, 26, 27, 37, 47 (0-based indices)
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "lowgamma": (30, 48), "highgamma": (52, 300)}

_cache = {}


def load_raw():
    if "y" not in _cache:
        _cache["y"] = sio.loadmat(DATA)["y"]
    y = _cache["y"]
    cue, glove = y[61], y[62:67]
    on = np.where((np.diff(cue) != 0) & (cue[1:] != 0))[0] + 1
    off = np.where((np.diff(cue) != 0) & (cue[1:] == 0))[0] + 1
    labels = cue[on].astype(int)
    return y, cue, glove, on, off, labels


def _filt(sos, x, causal):
    return ss.sosfilt(sos, x, axis=1) if causal else ss.sosfiltfilt(sos, x, axis=1)


def clean(causal=True):
    """CAR -> 1 Hz high-pass -> 50 Hz notches (+harmonics)."""
    key = ("clean", causal)
    if key in _cache:
        return _cache[key]
    y = load_raw()[0]
    e = y[1:61] - y[1:61].mean(axis=0)
    e = _filt(ss.butter(4, 1, "highpass", fs=fs, output="sos"), e, causal)
    for f0 in [50, 100, 150, 200, 250, 300]:
        b, a = ss.iirnotch(f0, Q=30, fs=fs)
        e = ss.lfilter(b, a, e, axis=1) if causal else ss.filtfilt(b, a, e, axis=1)
    _cache[key] = e
    return e


def band_power(band, causal=True, lo_hi=None):
    """Instantaneous power (squared band-passed signal), float32."""
    lo, hi = lo_hi if lo_hi else BANDS[band]
    key = ("bp", lo, hi, causal)
    if key not in _cache:
        e = clean(causal)
        x = _filt(ss.butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), e, causal)
        _cache[key] = (x ** 2).astype(np.float32)
    return _cache[key]


def win_log(P, start, t0, t1, chans=slice(None)):
    """log10 mean power of P[chans] between start+t0 and start+t1 seconds."""
    return np.log10(P[chans, start + int(round(t0 * fs)): start + int(round(t1 * fs))].mean(axis=1))


def decoder_features(hg, onsets, edges=np.arange(0.25, 2.26, 0.25)):
    """Original 480-feature set: log HG power, 60 ch x 8 bins of 250 ms."""
    return np.array([np.concatenate([win_log(hg, o, a, b) for a, b in zip(edges[:-1], edges[1:])])
                     for o in onsets])
