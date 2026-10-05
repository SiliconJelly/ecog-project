"""Signal-only processing. Cues and glove data cannot enter these interfaces."""
import time
import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz

FS = 1200
EDGES = np.arange(.25, 2.26, .25)
DEADLINES = [.5, .75, 1., 1.25, 1.5, 2., 2.25]
BUDGETS = [10, 20, 40, 60]


def reference(raw, montage="car", channel_ids=None):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[0] < 2 or not np.isfinite(raw).all():
        raise ValueError("ECoG must have at least two finite channels")
    if montage == "car":
        return raw - raw.mean(axis=0)
    if montage != "local":
        raise ValueError("Unknown montage")
    ids = np.arange(raw.shape[0]) if channel_ids is None else np.asarray(channel_ids)
    lookup = {int(c): i for i, c in enumerate(ids)}
    out = np.empty_like(raw)
    for i, c in enumerate(ids):
        row, col = int(c) % 10, int(c) // 10
        neighbors = [lookup[j] for j in ids if abs(int(j)%10-row)+abs(int(j)//10-col)==1]
        if not neighbors:
            raise ValueError("Local montage channel has no recorded grid neighbor")
        out[i] = raw[i] - raw[neighbors].mean(axis=0)
    return out


class Filter:
    """Forward filter with explicit state retained between sample chunks."""
    def __init__(self, channels, sos=None, b=None, a=None):
        self.sos, self.b, self.a = sos, b, a
        self.zi = (np.zeros((len(sos), channels, 2)) if sos is not None
                   else np.zeros((channels, max(len(b), len(a))-1)))

    def push(self, x):
        if self.sos is not None:
            y, self.zi = signal.sosfilt(self.sos, x, axis=1, zi=self.zi)
        else:
            y, self.zi = signal.lfilter(self.b, self.a, x, axis=1, zi=self.zi)
        return y


def clean(raw, montage="car", channel_ids=None, chunk=120):
    n = raw.shape[0]
    filters = [Filter(n, sos=signal.butter(4, 1, "highpass", fs=FS, output="sos"))]
    for hz in [50, 100, 150, 200, 250, 300]:
        b, a = signal.iirnotch(hz, 30, fs=FS)
        filters.append(Filter(n, b=b, a=a))
    out = np.empty(raw.shape, dtype=float)
    durations = []
    for start in range(0, raw.shape[1], chunk):
        begun = time.perf_counter()
        x = reference(raw[:, start:start+chunk], montage, channel_ids)
        for f in filters:
            x = f.push(x)
        out[:, start:start+chunk] = x
        durations.append(time.perf_counter()-begun)
    return out, {"chunk_samples": chunk, "chunk_input_seconds": chunk/FS,
                 "cleaning_compute_seconds": float(np.sum(durations)),
                 "cleaning_chunk_p50_ms": float(np.percentile(durations, 50)*1000),
                 "cleaning_chunk_p95_ms": float(np.percentile(durations, 95)*1000)}


def fit_ar(calibration, order=10):
    if calibration.ndim != 2 or calibration.shape[1] <= order or not np.isfinite(calibration).all():
        raise ValueError("Invalid pre-task AR calibration")
    coefficients = []
    for row in calibration:
        row = row - row.mean()
        r = np.array([row[k:] @ row[:len(row)-k] / len(row) for k in range(order+1)])
        if r[0] <= np.finfo(float).tiny:
            raise ValueError("Flat calibration channel; no AR model can be fitted")
        r /= r[0]
        a = solve_toeplitz(r[:-1], r[1:])
        if not np.isfinite(a).all() or 1-a@r[1:] <= 0:
            raise ValueError("Invalid AR prediction-error variance")
        coefficients.append(np.r_[1., -a])
    return np.asarray(coefficients)


def whiten(x, coefficients, chunk=120):
    out = np.empty_like(x)
    states = np.zeros((len(x), coefficients.shape[1]-1))
    for start in range(0, x.shape[1], chunk):
        for ch, a in enumerate(coefficients):
            out[ch, start:start+chunk], states[ch] = signal.lfilter(a, [1.], x[ch, start:start+chunk], zi=states[ch])
    return out


def band_filter(x, band, chunk=120):
    f = Filter(len(x), sos=signal.butter(4, band, "bandpass", fs=FS, output="sos"))
    out = np.empty_like(x)
    for start in range(0, x.shape[1], chunk):
        out[:, start:start+chunk] = f.push(x[:, start:start+chunk])
    return out


def power_features(filtered, onsets):
    power = filtered**2
    features = []
    for o in onsets:
        edges = o + (EDGES*FS).astype(int)
        if edges[-1] > power.shape[1] or edges[0] < 0:
            raise ValueError("Feature window outside recording")
        features.append(np.stack([power[:, a:b].mean(axis=1) for a,b in zip(edges[:-1],edges[1:])], axis=1))
    return np.log10(np.maximum(features, np.finfo(float).tiny))


def build_features(raw, onsets, montage="car", channel_ids=None, chunk=120, return_signals=False):
    started = time.perf_counter()
    cleaned, timing = clean(raw, montage, channel_ids, chunk)
    fit_end = int(onsets[0]-2*FS)
    if fit_end <= 2*FS+10:
        raise ValueError("Insufficient cue-free pre-task calibration")
    coefficients = fit_ar(cleaned[:, 2*FS:fit_end])
    # Large chunks for the per-channel FIR reduce Python overhead; state is still retained.
    processed = whiten(cleaned, coefficients, chunk=12000)
    hg = band_filter(processed, [50,300], chunk=chunk)
    features = power_features(hg, onsets)
    timing["pipeline_wall_seconds"] = time.perf_counter()-started
    timing["ar_fit_seconds"] = [2., fit_end/FS]
    if return_signals:
        return features, coefficients, timing, cleaned, hg
    return features, coefficients, timing
