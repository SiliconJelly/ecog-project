"""Separate, reproducible ECoG experiments; never modifies the team submission.

Run: python3 explore_ecog_angles.py
Requires numpy, scipy, scikit-learn, matplotlib, threadpoolctl.
All predictions use ECoG alone; cues provide trial timing and evaluation labels.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat
from scipy import signal
from scipy.linalg import solve_toeplitz
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/contribution_angles"
FS = 1200
EDGES = np.arange(0.25, 2.26, 0.25)
NAMES = {1: "Rock", 2: "Scissors", 3: "Paper"}
CANDIDATES = ["hg_lda", "relative_hg_lda", "hg_beta_lda", "top10_lda", "top20_lda", "hg_svm"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_causal(raw):
    x = raw - raw.mean(axis=0)
    x = signal.sosfilt(signal.butter(4, 1, "highpass", fs=FS, output="sos"), x, axis=1)
    for hz in [50, 100, 150, 200, 250, 300]:
        b, a = signal.iirnotch(hz, Q=30, fs=FS)
        x = signal.lfilter(b, a, x, axis=1)
    return x


def fit_ar(calibration, order=10):
    coefficients = []
    for row in calibration:
        row = row - row.mean()
        r = np.array([row[k:] @ row[:len(row) - k] / len(row) for k in range(order + 1)])
        if r[0] <= 0:
            raise ValueError("Zero-variance calibration channel")
        r /= r[0]
        a = solve_toeplitz(r[:-1], r[1:])
        if 1 - a @ r[1:] <= 0:
            raise ValueError("Invalid AR residual variance")
        coefficients.append(np.r_[1, -a])
    return np.array(coefficients)


def whiten(x, coefficients):
    return np.stack([signal.lfilter(a, [1], row) for row, a in zip(x, coefficients)])


def log_features(x, onsets, band):
    x = signal.sosfilt(signal.butter(4, band, "bandpass", fs=FS, output="sos"), x, axis=1)
    x **= 2
    features, baselines = [], []
    for onset in onsets:
        edges = onset + (EDGES * FS).astype(int)
        if onset < FS or edges[-1] > x.shape[1]:
            raise ValueError("Trial outside recording")
        features.append(np.stack([x[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1))
        baselines.append(x[:, onset-FS:onset].mean(axis=1))
    features = np.log10(np.maximum(features, np.finfo(float).tiny))
    baselines = np.log10(np.maximum(baselines, np.finfo(float).tiny))
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    return features, features - baselines[:, :, None]


def blocked_splits(indices, folds=5, purge=1):
    """Contiguous test blocks; purge neighboring trial IDs from training."""
    indices = np.sort(np.asarray(indices))
    for test in np.array_split(indices, folds):
        excluded = np.concatenate([test + delta for delta in range(-purge, purge+1)])
        train = indices[~np.isin(indices, excluded)]
        if np.intersect1d(train, test).size or np.any(np.abs(train[:, None] - test).min(axis=1) <= purge):
            raise AssertionError("Purged split overlaps")
        yield train, test


def fit_predict(bank, labels, train, test, candidate="hg_lda", bins=8, channels=60):
    x = bank["relative_hg" if candidate == "relative_hg_lda" else "hg"][:, :, :bins]
    if candidate.startswith("top"):
        channels = int(candidate.split("_")[0][3:])
    selected = np.arange(60)
    if channels < 60:
        # Score all bins using training labels only, then keep complete channels.
        f, _ = f_classif(x[train].reshape(len(train), -1), labels[train])
        score = np.nan_to_num(f, nan=0, posinf=0).reshape(60, bins).mean(axis=1)
        selected = np.argsort(-score, kind="stable")[:channels]
    x = x[:, selected].reshape(len(labels), -1)
    if candidate == "hg_beta_lda":
        x = np.concatenate([x, bank["beta"][:, :, :bins].reshape(len(labels), -1)], axis=1)
    model = (make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)) if candidate == "hg_svm"
             else LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if set(labels[train]) != {1, 2, 3}:
        raise ValueError("Training split lacks a class")
    model.fit(x[train], labels[train])
    return model.predict(x[test]), selected


def summarize(labels, predicted, scores):
    predicted = np.asarray(predicted)
    if predicted.ndim == 1:
        predicted = predicted[None, :]
    truth = np.tile(labels, predicted.shape[0])
    flat = predicted.ravel()
    repeats = (predicted == labels).mean(axis=1) * 100
    return {"accuracy_percent": float(accuracy_score(truth, flat) * 100),
            "macro_f1_percent": float(f1_score(truth, flat, labels=[1, 2, 3], average="macro") * 100),
            "recall_percent": dict(zip(NAMES.values(), (recall_score(truth, flat, labels=[1, 2, 3], average=None) * 100).tolist())),
            "fold_sd_percent": float(np.std(scores) * 100),
            "per_repeat_accuracy_percent": repeats.tolist(),
            "confusion_matrix_counts": confusion_matrix(truth, flat, labels=[1, 2, 3]).tolist(),
            "prediction_count": int(flat.size), "unique_trials": len(labels)}


def evaluate(bank, labels, splits, candidate="hg_lda", bins=8, channels=60, repeats=1):
    predicted = np.full((repeats, len(labels)), -1, dtype=int)
    coverage = np.zeros_like(predicted)
    scores, selections = [], []
    folds = len(splits) // repeats
    for i, (train, test) in enumerate(splits):
        pred, selected = fit_predict(bank, labels, train, test, candidate, bins, channels)
        repeat = i // folds
        predicted[repeat, test] = pred
        coverage[repeat, test] += 1
        scores.append(float(np.mean(pred == labels[test])))
        selections.append(selected.tolist())
    if not np.all(coverage == 1):
        raise AssertionError("Every trial must be tested once per repeat")
    result = summarize(labels, predicted, scores)
    result["selected_channels_1_based_per_fold"] = [[j+1 for j in s] for s in selections]
    return result, predicted, np.array(scores)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / "source/ECoG_Handpose.mat", ROOT / "source/classify.py",
                 ROOT / "team_submission_whole_record/classify_whitened.py"]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    data = loadmat(protected[0])["y"]
    raw, cue = data[1:61], data[61]
    onsets = np.flatnonzero((np.diff(cue) != 0) & (cue[1:] != 0)) + 1
    labels = cue[onsets].astype(int)
    if data.shape != (67, 507025) or not np.array_equal(np.bincount(labels)[1:], [30, 30, 30]):
        raise ValueError("Unexpected dataset")
    random_splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0).split(np.zeros((90, 1)), labels))
    temporal_splits = list(blocked_splits(np.arange(90)))
    arrays = {"labels": labels, "onsets": onsets}
    results = {"offline_saved": {}, "causal_candidates": {}, "latency": {}, "channel_budget": {}}
    print("Checking saved offline results and temporal sensitivity...", flush=True)
    for key, path, feature_key, score_key in [
        ("original", ROOT/"team_submission_whole_record/results/off/arrays.npz", "features", "three_class_scores"),
        ("whole_record_ar", ROOT/"team_submission_whole_record/results/on/arrays.npz", "features", "three_class_scores"),
        ("initial_rest_ar", ROOT/"results/whitening/comparison_arrays.npz", "whitening_features", "whitening_scores")]:
        with np.load(path) as z:
            np.testing.assert_array_equal(labels, z["labels"])
            bank = {"hg": z[feature_key].reshape(90, 60, 8)}
            rr, rp, rs = evaluate(bank, labels, random_splits, repeats=10)
            np.testing.assert_allclose(rs, z[score_key], atol=1e-12)
            bb, bp, _ = evaluate(bank, labels, temporal_splits)
        results["offline_saved"][key] = {"repeated_cv": rr, "purged_block_cv": bb}
        arrays[key+"_random_predictions"] = rp
        arrays[key+"_blocked_predictions"] = bp
    print("Building causal features; AR fitted before the first cue...", flush=True)
    cleaned = clean_causal(raw)
    fit_start, fit_end = 2*FS, int(onsets[0]-2*FS)
    coefficients = fit_ar(cleaned[:, fit_start:fit_end])
    processed = whiten(cleaned, coefficients)
    hg, relative = log_features(processed, onsets, [50, 300])
    # Beta is taken before whitening, avoiding unnecessary suppression of slow rhythms.
    beta, _ = log_features(cleaned, onsets, [13, 30])
    bank = {"hg": hg, "relative_hg": relative, "beta": beta}
    # Truncating future samples must not change already available features.
    cutoff = int(onsets[0] + 0.75*FS)
    hp_sos = signal.butter(4, [50, 300], "bandpass", fs=FS, output="sos")
    prefix_filtered = signal.sosfilt(hp_sos, whiten(clean_causal(raw[:, :cutoff]), coefficients), axis=1)
    full_filtered = signal.sosfilt(hp_sos, processed[:, :cutoff], axis=1)
    np.testing.assert_array_equal(prefix_filtered, full_filtered)
    arrays.update(causal_hg=hg, causal_relative_hg=relative, causal_beta=beta, causal_ar_coefficients=coefficients)
    del data, raw, cleaned, processed, full_filtered, prefix_filtered
    for candidate in CANDIDATES:
        print("Evaluating causal candidate:", candidate, flush=True)
        rr, rp, _ = evaluate(bank, labels, random_splits, candidate, repeats=10)
        bb, bp, _ = evaluate(bank, labels, temporal_splits, candidate)
        results["causal_candidates"][candidate] = {"repeated_cv": rr, "purged_block_cv": bb}
        arrays[candidate+"_random_predictions"] = rp
        arrays[candidate+"_blocked_predictions"] = bp
    for bins in [1, 2, 3, 4, 5, 7, 8]:
        deadline = float(EDGES[bins])
        print("Decision deadline:", deadline, "s after cue", flush=True)
        rr, rp, _ = evaluate(bank, labels, random_splits, bins=bins, repeats=10)
        bb, bp, _ = evaluate(bank, labels, temporal_splits, bins=bins)
        results["latency"][str(deadline)] = {"repeated_cv": rr, "purged_block_cv": bb}
        arrays[f"deadline_{deadline}_random_predictions"] = rp
        arrays[f"deadline_{deadline}_blocked_predictions"] = bp
    for channels in [5, 10, 20, 40, 60]:
        print("Selected feature channels:", channels, flush=True)
        rr, rp, _ = evaluate(bank, labels, random_splits, channels=channels, repeats=10)
        bb, bp, _ = evaluate(bank, labels, temporal_splits, channels=channels)
        results["channel_budget"][str(channels)] = {"repeated_cv": rr, "purged_block_cv": bb}
        arrays[f"channels_{channels}_random_predictions"] = rp
    print("Nested model comparison: five outer blocks, three inner blocks...", flush=True)
    nested_pred = np.full(90, -1, dtype=int)
    nested_folds = []
    for outer_i, (train, test) in enumerate(temporal_splits):
        inner = list(blocked_splits(train, folds=3))
        means = {}
        for candidate in CANDIDATES:
            correct, total = 0, 0
            for inner_train, inner_test in inner:
                pred, _ = fit_predict(bank, labels, inner_train, inner_test, candidate)
                correct += int(np.sum(pred == labels[inner_test]))
                total += len(inner_test)
            means[candidate] = correct/total
        chosen = max(CANDIDATES, key=lambda k: means[k])  # Fixed order breaks ties.
        pred, _ = fit_predict(bank, labels, train, test, chosen)
        nested_pred[test] = pred
        nested_folds.append({"outer_fold": outer_i+1, "train_trials_1_based": (train+1).tolist(),
                             "test_trials_1_based": (test+1).tolist(), "chosen_candidate": chosen,
                             "inner_accuracy_percent": {k:v*100 for k,v in means.items()},
                             "outer_accuracy_percent": float(np.mean(pred==labels[test])*100)})
    if np.any(nested_pred < 0):
        raise AssertionError("Incomplete nested predictions")
    results["nested_purged_block_cv"] = summarize(labels, nested_pred, [r["outer_accuracy_percent"]/100 for r in nested_folds])
    results["nested_purged_block_cv"]["folds"] = nested_folds
    arrays["nested_predictions"] = nested_pred
    print("Forward evaluation: train on earlier trials, test the next 15...", flush=True)
    results["forward_causal_hg"] = []
    for test_start in [45, 60, 75]:
        train, test = np.arange(test_start-1), np.arange(test_start, test_start+15)
        pred, _ = fit_predict(bank, labels, train, test)
        results["forward_causal_hg"].append({"train_trials": len(train), "test_trials_1_based": (test+1).tolist(),
                                            "correct": int(np.sum(pred==labels[test])), "total": len(test),
                                            "accuracy_percent": float(np.mean(pred==labels[test])*100)})
        arrays[f"forward_{test_start}_predictions"] = pred
    print("Label-shuffle control for causal high-gamma (20 shuffles)...", flush=True)
    rng = np.random.default_rng(0)
    shuffled = []
    for _ in range(20):
        yy = rng.permutation(labels)
        _, pred, _ = evaluate(bank, yy, temporal_splits)
        shuffled.append(float(np.mean(pred==yy)))
    results["causal_hg_blocked_shuffled_mean_percent"] = float(np.mean(shuffled)*100)
    arrays["shuffled_blocked_accuracies"] = np.array(shuffled)
    for i, (train, test) in enumerate(temporal_splits):
        arrays[f"block_{i}_train"] = train
        arrays[f"block_{i}_test"] = test
    # A CSV enables a replay demo using genuinely held-out model predictions.
    pred = arrays["hg_lda_random_predictions"]
    with (OUT/"trial_predictions.csv").open("w") as f:
        f.write("trial,cue,causal_pred_repeat1,correct_repeat1,error_fraction_10_repeats\n")
        for i in range(90):
            f.write(f"{i+1},{NAMES[labels[i]]},{NAMES[pred[0,i]]},{int(pred[0,i]==labels[i])},{np.mean(pred[:,i]!=labels[i]):.2f}\n")
    results["provenance"] = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_and_reference_sha256": before, "script_sha256": sha256(__file__),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "blas_threads": 1, "ar_fit_seconds": [fit_start/FS, fit_end/FS],
        "trial_count": 90, "random_cv": "10 folds x 10 repeats, seed 0",
        "blocked_cv": "5 contiguous blocks of 18 trials; purge one adjacent trial on each side",
        "nested_cv": "Same outer blocks; 3 inner contiguous blocks with one-trial purge; select six fixed candidates",
        "causal_future_truncation_check_passed": True, "protected_files_unchanged": all(sha256(p)==before[str(p.relative_to(ROOT))] for p in protected),
        "elapsed_seconds": time.perf_counter()-started}
    results["limitations"] = [
        "Exploratory reanalysis of an already examined recording, not a new blinded test or new subject.",
        "Random and blocked CV train on both earlier and later trials; forward evaluation alone trains only on the past.",
        "Offline saved features use zero-phase filters; whole-record AR also fits on test-period signal values.",
        "Causal replay is cue-triggered; decision deadlines are sample availability, not a measured deployed latency.",
        "Channel selection reduces classifier inputs; CAR still uses all 60 recording channels.",
        "Repeated predictions are not independent trials; fold SD is not a confidence interval.",
        "Candidate and curve comparisons are descriptive. Selecting the best outer score would bias a performance claim.",
        "Nested CV assesses the specified selection procedure; prior human exploration still limits independence.",
        "Cue labels are requested gestures. No glove channels enter the decoder; labels were not corrected post hoc.",
        "A chance-level shuffle is a sanity check; it cannot rule out temporal dependence or stimulus-related confounds."]
    np.savez_compressed(OUT/"experiment_arrays.npz", **arrays)
    (OUT/"metrics.json").write_text(json.dumps(results, indent=2)+"\n")
    draw_figures(results)
    print("Finished:", OUT, flush=True)
    for c, v in results["causal_candidates"].items():
        print(c, "random", round(v["repeated_cv"]["accuracy_percent"],2), "blocked", round(v["purged_block_cv"]["accuracy_percent"],2))
    print("Nested blocked:", results["nested_purged_block_cv"]["accuracy_percent"])


def draw_figures(results):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for protocol, color, label in [("repeated_cv", "#2778b1", "Random trial CV (10 x 10)"),
                                   ("purged_block_cv", "#d36937", "Purged temporal blocks (5)")]:
        xx = [float(x) for x in results["latency"]]
        yy = [v[protocol]["accuracy_percent"] for v in results["latency"].values()]
        ax[0].plot(xx, yy, "o-", color=color, label=label)
        xx = [int(x) for x in results["channel_budget"]]
        yy = [v[protocol]["accuracy_percent"] for v in results["channel_budget"].values()]
        ax[1].plot(xx, yy, "o-", color=color, label=label)
    for a in ax:
        a.set_ylim(70, 101)
        a.grid(alpha=0.2)
        a.set_ylabel("Accuracy (%)")
    ax[0].set(xlabel="Feature window ends at this time after cue (s)", title="How soon can ECoG classify the gesture?")
    ax[1].set(xlabel="Number of channels used by the classifier", title="How many feature channels are needed?")
    ax[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Causal filters + pre-task AR calibration | 90 trials, one recording", fontsize=12)
    fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.9))
    fig.text(0.02, 0.035, "Exploratory curves; no confidence intervals. Channel selection uses training labels only; CAR still uses all 60 electrodes.", fontsize=9)
    fig.savefig(OUT/"speed_and_channels.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    labels = ["Offline original", "Offline whole-record AR", "Offline pre-task AR", "Causal pre-task AR", "Causal nested selection"]
    values = [results["offline_saved"][k]["purged_block_cv"]["accuracy_percent"] for k in ["original", "whole_record_ar", "initial_rest_ar"]]
    values += [results["causal_candidates"]["hg_lda"]["purged_block_cv"]["accuracy_percent"], results["nested_purged_block_cv"]["accuracy_percent"]]
    ax.barh(labels, values, color=["#999999", "#5599b4", "#5599b4", "#d36937", "#658969"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 105), xlabel="Accuracy (%)", title="A stricter question: can the decoder classify a held-out time block?")
    for i, v in enumerate(values):
        ax.text(v+0.7, i, f"{v:.1f}%", va="center")
    ax.axvline(100/3, color="#666666", linestyle="--", linewidth=1)
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.86))
    fig.text(0.02, 0.035, "Five chronological blocks; one neighboring trial purged. Offline features include future samples; causal features do not.\nNested selection occurs inside training folds. These are within-recording exploratory estimates, not new-session accuracy.", fontsize=9)
    fig.savefig(OUT/"validation_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
