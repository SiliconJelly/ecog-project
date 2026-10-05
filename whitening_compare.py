"""Compare Abri's baseline with AR(10) whitening from pre-task calibration."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.linalg import solve_toeplitz, toeplitz
from scipy import signal
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from threadpoolctl import threadpool_limits


SAMPLING_RATE = 1200
AR_ORDER = 10
CALIBRATION_GUARD_SECONDS = 2.0
BIN_EDGES_SECONDS = np.arange(0.25, 2.26, 0.25)
CLASS_NAMES = ["Fist", "Peace", "Open"]
PROJECT_DIRECTORY = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_signal(raw_signal):
    """Apply the same CAR, high-pass, and notch steps as classify.py."""
    cleaned = raw_signal - raw_signal.mean(axis=0)
    highpass = signal.butter(4, 1, "highpass", fs=SAMPLING_RATE, output="sos")
    cleaned = signal.sosfiltfilt(highpass, cleaned, axis=1)
    for notch_frequency in [50, 100, 150, 200, 250, 300]:
        numerator, denominator = signal.iirnotch(notch_frequency, Q=30, fs=SAMPLING_RATE)
        cleaned = signal.filtfilt(numerator, denominator, cleaned, axis=1)
    return cleaned


def fit_ar_whitening(calibration, order=AR_ORDER):
    """Return channel-wise prediction-error FIR filters using biased Yule-Walker."""
    calibration = np.asarray(calibration, dtype=np.float64)
    if calibration.ndim != 2 or calibration.shape[1] <= order or order < 1:
        raise ValueError("Calibration must contain channels and more samples than AR order.")
    if not np.isfinite(calibration).all():
        raise ValueError("Calibration contains non-finite values.")
    coefficients = np.empty((calibration.shape[0], order + 1))
    residual_variance_ratios = []
    for channel_index, channel_signal in enumerate(calibration):
        centered = channel_signal - channel_signal.mean()
        sample_count = centered.size
        covariance = np.array([
            np.dot(centered[:sample_count - lag], centered[lag:]) / sample_count
            for lag in range(order + 1)
        ])
        if covariance[0] <= np.finfo(float).tiny:
            raise ValueError(f"Calibration channel {channel_index + 1} has no variance.")
        covariance /= covariance[0]
        predictor = solve_toeplitz(covariance[:-1], covariance[1:])
        np.testing.assert_allclose(
            toeplitz(covariance[:-1]) @ predictor, covariance[1:], rtol=1e-7, atol=1e-9
        )
        residual_ratio = 1.0 - predictor @ covariance[1:]
        if residual_ratio <= 0 or not np.isfinite(residual_ratio):
            raise ValueError(f"Invalid prediction-error variance in channel {channel_index + 1}.")
        coefficients[channel_index] = np.r_[1.0, -predictor]
        residual_variance_ratios.append(residual_ratio)
    return coefficients, np.asarray(residual_variance_ratios)


def apply_whitening(cleaned, coefficients):
    if coefficients.shape[0] != cleaned.shape[0]:
        raise ValueError("The filter and signal channel counts differ.")
    whitened = np.empty_like(cleaned)
    for channel_index, channel_coefficients in enumerate(coefficients):
        whitened[channel_index] = signal.lfilter(
            channel_coefficients, [1.0], cleaned[channel_index]
        )
    if not np.isfinite(whitened).all():
        raise ValueError("Whitening produced non-finite values.")
    return whitened


def extract_features(processed_signal, onsets):
    """Match classify.py: 60 channels by eight 250 ms log-power bins."""
    bandpass = signal.butter(4, [50, 300], "bandpass", fs=SAMPLING_RATE, output="sos")
    high_gamma = signal.sosfiltfilt(bandpass, processed_signal, axis=1)
    high_gamma **= 2
    features = []
    for onset in onsets:
        edges = (onset + BIN_EDGES_SECONDS * SAMPLING_RATE).astype(int)
        if edges[0] < 0 or edges[-1] > processed_signal.shape[1]:
            raise ValueError("A feature window exceeds the recording boundaries.")
        powers = [
            high_gamma[:, start:stop].mean(axis=1)
            for start, stop in zip(edges[:-1], edges[1:])
        ]
        features.append(np.log10(np.stack(powers, axis=1)).ravel())
    features = np.asarray(features)
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return features


def evaluate_splits(features, labels, splits, fold_count=10):
    predictions = np.full((len(splits) // fold_count, labels.size), -1, dtype=int)
    coverage = np.zeros_like(predictions)
    scores = []
    for split_index, (train_indices, test_indices) in enumerate(splits):
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError("Training and test trials overlap within a fold.")
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        classifier.fit(features[train_indices], labels[train_indices])
        predicted = classifier.predict(features[test_indices])
        scores.append(np.mean(predicted == labels[test_indices]))
        repetition = split_index // fold_count
        predictions[repetition, test_indices] = predicted
        coverage[repetition, test_indices] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each trial must be tested once in each repetition.")
    return np.asarray(scores), predictions


def shuffled_control(features, labels):
    random = np.random.default_rng(0)
    scores = []
    for repeat_index in range(20):
        shuffled = random.permutation(labels)
        splits = list(StratifiedKFold(10, shuffle=True, random_state=repeat_index).split(features, shuffled))
        fold_scores, _ = evaluate_splits(features, shuffled, splits)
        scores.append(fold_scores.mean())
    return np.asarray(scores)


def summarize_scores(scores):
    return {
        "mean_percent": float(scores.mean() * 100),
        "std_percent": float(scores.std() * 100),
        "sample_count": int(scores.size),
    }


def set_plot_style():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for family in ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Arial Unicode MS"]:
        if family in available_fonts:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "figure.dpi": 120})


def plot_spectra(cleaned, whitened, output_directory):
    before_spectra = []
    after_spectra = []
    for before_channel, after_channel in zip(cleaned, whitened):
        frequencies, before_channel_psd = signal.welch(
            before_channel[2 * SAMPLING_RATE:], fs=SAMPLING_RATE, nperseg=2400
        )
        _, after_channel_psd = signal.welch(
            after_channel[2 * SAMPLING_RATE:], fs=SAMPLING_RATE, nperseg=2400
        )
        before_spectra.append(before_channel_psd)
        after_spectra.append(after_channel_psd)
    before_psd = np.asarray(before_spectra)
    after_psd = np.asarray(after_spectra)
    frequency_mask = (frequencies >= 50) & (frequencies <= 300)
    for harmonic in [50, 100, 150, 200, 250, 300]:
        frequency_mask &= np.abs(frequencies - harmonic) > 3
    normalized_before = before_psd / before_psd[:, frequency_mask].mean(axis=1, keepdims=True)
    normalized_after = after_psd / after_psd[:, frequency_mask].mean(axis=1, keepdims=True)
    median_before = np.median(10 * np.log10(normalized_before), axis=0)
    median_after = np.median(10 * np.log10(normalized_after), axis=0)
    figure, axis = plt.subplots(figsize=(10, 5.4))
    visible = (frequencies >= 10) & (frequencies <= 350)
    axis.plot(frequencies[visible], median_before[visible], label="기존 전처리", color="#235789", linewidth=2)
    axis.plot(frequencies[visible], median_after[visible], label="Whitening 추가", color="#df6c28", linewidth=2)
    axis.axvspan(50, 300, color="#dddddd", alpha=0.25)
    axis.set(xlabel="주파수 (Hz): 오른쪽으로 갈수록 빠른 진동", ylabel="정규화한 신호 파워 (dB)",
             title="Whitening 전후: 주파수별 신호 크기의 분포", xlim=(10, 350))
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.text(0.5, 0.02, "60채널 중앙값 · 각 곡선을 50–300 Hz 평균에 맞춤(notch 주변 제외) · 회색: 분류 대역",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output_directory / "whitening_spectrum.png", dpi=180)
    plt.close(figure)
    flatness_before = np.exp(np.log(before_psd[:, frequency_mask]).mean(axis=1)) / before_psd[:, frequency_mask].mean(axis=1)
    flatness_after = np.exp(np.log(after_psd[:, frequency_mask]).mean(axis=1)) / after_psd[:, frequency_mask].mean(axis=1)
    return {
        "frequencies": frequencies,
        "normalized_median_before_db": median_before,
        "normalized_median_after_db": median_after,
        "flatness_before": flatness_before,
        "flatness_after": flatness_after,
    }


def plot_comparison(baseline_scores, whitened_scores, baseline_matrix, whitened_matrix, output_directory):
    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    means = [baseline_scores.mean() * 100, whitened_scores.mean() * 100]
    deviations = [baseline_scores.std() * 100, whitened_scores.std() * 100]
    axis.bar([0, 1], means, yerr=deviations, capsize=7, width=0.55, color=["#235789", "#df6c28"])
    axis.set_xticks([0, 1], ["기존 코드", "Whitening 추가"])
    axis.set(ylabel="평균 정확도 (%)", ylim=(0, 112), title="같은 학습·시험 분할에서 비교")
    axis.axhline(100 / 3, color="#777777", linestyle="--", linewidth=1, label="무작위 추측 기준 33.3%")
    for condition_index, mean in enumerate(means):
        axis.text(condition_index, mean - 15, f"{mean:.1f}%", ha="center", color="white", fontsize=18)
    axis.legend(loc="lower right", frameon=False)
    axis.grid(axis="y", alpha=0.15)
    axis.set_axisbelow(True)
    figure.text(0.5, 0.02, "10-fold × 10회 반복 · 오차막대는 100개 시험 묶음의 표준편차이며 신뢰구간이 아님", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output_directory / "accuracy_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    for axis, matrix, condition in zip(axes, [baseline_matrix, whitened_matrix], ["기존 코드", "Whitening 추가"]):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=30)
        for row_index in range(3):
            for column_index in range(3):
                count = int(matrix[row_index, column_index])
                axis.text(column_index, row_index, str(count), ha="center", va="center", fontsize=17,
                          color="white" if count > 15 else "#18334c")
        axis.set_xticks(range(3), ["바위", "가위", "보"])
        axis.set_yticks(range(3), ["바위", "가위", "보"])
        axis.set(xlabel="프로그램의 예측", ylabel="실제 동작", title=f"{condition}: {int(np.trace(matrix))}/90 정답")
    figure.suptitle("첫 번째 10-fold 평가의 예측 결과", fontsize=15)
    figure.text(0.5, 0.02, "두 그림은 동일한 시험 기록에 대한 예측 · 전체 반복 평가의 평균과는 구별", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output_directory / "confusion_comparison.png", dpi=180)
    plt.close(figure)


def main():
    started = time.perf_counter()
    data_path = PROJECT_DIRECTORY / "source/ECoG_Handpose.mat"
    baseline_path = PROJECT_DIRECTORY / "results/baseline_results.npz"
    original_script = PROJECT_DIRECTORY / "source/classify.py"
    source_hash_before = file_sha256(original_script)
    output_directory = PROJECT_DIRECTORY / "results/whitening"
    output_directory.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    with np.load(baseline_path) as saved:
        baseline = {name: saved[name].copy() for name in saved.files}
    print("1/6 Loading data and choosing cue-free calibration before any trial.", flush=True)
    record = loadmat(data_path, variable_names=["y"])["y"]
    if record.shape[0] != 67 or not np.isfinite(record).all():
        raise ValueError("Unexpected recording dimensions or non-finite values.")
    times = record[0].copy()
    cue = record[61].copy()
    raw_signal = np.array(record[1:61], order="C", copy=True)
    del record
    np.testing.assert_allclose(np.diff(times), 1 / SAMPLING_RATE, rtol=1e-7, atol=1e-12)
    onsets = np.flatnonzero((np.diff(cue) != 0) & (cue[1:] != 0)) + 1
    labels = cue[onsets].astype(int)
    np.testing.assert_array_equal(labels, baseline["labels"])
    np.testing.assert_array_equal(onsets, baseline["onsets"])
    first_onset = int(onsets[0])
    guard_samples = int(CALIBRATION_GUARD_SECONDS * SAMPLING_RATE)
    calibration_stop = first_onset - guard_samples
    if not np.all(cue[:first_onset] == 0) or calibration_stop - guard_samples < 5 * SAMPLING_RATE:
        raise ValueError("At least five seconds of guarded, pre-task calibration are required.")
    calibration_cleaned = clean_signal(raw_signal[:, :first_onset])
    coefficients, variance_ratios = fit_ar_whitening(calibration_cleaned[:, guard_samples:calibration_stop])
    del calibration_cleaned
    print(f"  AR({AR_ORDER}) fitted on {guard_samples / SAMPLING_RATE:.2f}–{calibration_stop / SAMPLING_RATE:.2f} s; first cue {first_onset / SAMPLING_RATE:.2f} s.", flush=True)

    print("2/6 Matching the unchanged baseline preprocessing and features.", flush=True)
    cleaned = clean_signal(raw_signal)
    del raw_signal
    baseline_features = extract_features(cleaned, onsets)
    feature_difference = float(np.max(np.abs(baseline_features - baseline["features"])))
    np.testing.assert_allclose(baseline_features, baseline["features"], rtol=1e-9, atol=1e-9)
    print(f"  Baseline feature maximum difference: {feature_difference:.3e}", flush=True)

    print("3/6 Applying fixed whitening, plotting spectra, and extracting the same features.", flush=True)
    whitened = apply_whitening(cleaned, coefficients)
    spectra = plot_spectra(cleaned, whitened, output_directory)
    del cleaned
    whitened_features = extract_features(whitened, onsets)
    del whitened

    print("4/6 Evaluating both conditions with identical 10-fold × 10 splits.", flush=True)
    splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0).split(baseline_features, labels))
    baseline_scores, baseline_predictions = evaluate_splits(baseline_features, labels, splits)
    np.testing.assert_array_equal(baseline_scores, baseline["accuracy"])
    np.testing.assert_array_equal(baseline_predictions[0], baseline["predictions"])
    whitened_scores, whitened_predictions = evaluate_splits(whitened_features, labels, splits)
    print(f"  Baseline {baseline_scores.mean() * 100:.2f}%; whitening {whitened_scores.mean() * 100:.2f}%.", flush=True)

    print("5/6 Fist/peace evaluation and the same 20 shuffled-label controls.", flush=True)
    fist_peace_mask = labels != 3
    fist_peace_labels = labels[fist_peace_mask]
    fist_peace_features = whitened_features[fist_peace_mask]
    fist_peace_splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0).split(fist_peace_features, fist_peace_labels))
    fist_peace_scores, _ = evaluate_splits(fist_peace_features, fist_peace_labels, fist_peace_splits)
    shuffled_scores = shuffled_control(whitened_features, labels)

    print("6/6 Saving figures, split indices, filter coefficients, predictions, and provenance.", flush=True)
    baseline_matrix = confusion_matrix(labels, baseline_predictions[0], labels=[1, 2, 3])
    whitened_matrix = confusion_matrix(labels, whitened_predictions[0], labels=[1, 2, 3])
    plot_comparison(baseline_scores, whitened_scores, baseline_matrix, whitened_matrix, output_directory)
    baseline_correct = baseline_predictions[0] == labels
    whitened_correct = whitened_predictions[0] == labels
    source_hash_after = file_sha256(original_script)
    if source_hash_before != source_hash_after:
        raise ValueError("The reference script changed during analysis.")
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "three_class": summarize_scores(baseline_scores),
            "fist_peace": summarize_scores(baseline["accuracy_fist_peace"]),
            "shuffled_control": summarize_scores(baseline["accuracy_shuffled"]),
            "confusion_matrix_first_repeat": baseline_matrix.tolist(),
        },
        "whitening": {
            "three_class": summarize_scores(whitened_scores),
            "fist_peace": summarize_scores(fist_peace_scores),
            "shuffled_control": summarize_scores(shuffled_scores),
            "confusion_matrix_first_repeat": whitened_matrix.tolist(),
        },
        "paired_difference_pp": float((whitened_scores - baseline_scores).mean() * 100),
        "per_repeat_accuracy_percent": {
            "baseline": (baseline_scores.reshape(10, 10).mean(axis=1) * 100).tolist(),
            "whitening": (whitened_scores.reshape(10, 10).mean(axis=1) * 100).tolist(),
        },
        "first_repeat_prediction_changes": {
            "corrected": int(np.sum(~baseline_correct & whitened_correct)),
            "new_errors": int(np.sum(baseline_correct & ~whitened_correct)),
        },
        "calibration": {
            "source": "Cue-free recording before the first task; processed separately from all task data",
            "raw_interval_seconds": [0, first_onset / SAMPLING_RATE],
            "fit_interval_seconds": [guard_samples / SAMPLING_RATE, calibration_stop / SAMPLING_RATE],
            "samples_used": calibration_stop - guard_samples,
            "ar_order": AR_ORDER,
            "estimator": "Biased, demeaned Yule-Walker; per-channel FIR coefficients [1, -predictor]",
            "application": "One-pass causal FIR after notch and before the unchanged high-gamma bandpass",
            "fit_uses_task_trials": False,
        },
        "evaluation": {
            "trial_count": int(labels.size), "class_counts": np.unique(labels, return_counts=True)[1].tolist(),
            "feature_shape": list(whitened_features.shape), "n_splits": 10, "n_repeats": 10,
            "random_state": 0, "identical_splits": True, "shuffled_repetitions": 20,
            "std_definition": "Population SD over 100 fold accuracies; not a confidence interval",
            "baseline_feature_max_abs_difference": feature_difference,
            "baseline_scores_exactly_reproduced": True,
            "baseline_other_controls_reused_from_verified_original_run": True,
        },
        "spectral_flatness_median_50_300_excluding_notches": {
            "baseline": float(np.median(spectra["flatness_before"])),
            "whitening": float(np.median(spectra["flatness_after"])),
        },
        "provenance": {
            "data_sha256": file_sha256(data_path), "reference_script_sha256": source_hash_after,
            "analysis_script_sha256": file_sha256(Path(__file__)),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "blas_threads": 1,
        },
        "limitations": [
            "One recording and 90 task trials; no independent participant or session test.",
            "Fixed zero-phase high-pass/notch/bandpass steps from the baseline are retained for offline comparability.",
            "Repeated CV folds share data and are not independent experiments.",
            "This is an AR-whitening ablation on Abri's LDA baseline, not a reproduction of the complete TVLDA paper.",
            "Calibration choice, AR order, feature windows, and split seed were fixed before examining whitening accuracy.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        output_directory / "comparison_arrays.npz",
        baseline_features=baseline_features, whitening_features=whitened_features, labels=labels, onsets=onsets,
        baseline_scores=baseline_scores, whitening_scores=whitened_scores,
        baseline_predictions=baseline_predictions, whitening_predictions=whitened_predictions,
        whitening_fist_peace_scores=fist_peace_scores, whitening_shuffled_scores=shuffled_scores,
        baseline_shuffled_scores=baseline["accuracy_shuffled"],
        train_indices=np.stack([indices for indices, _ in splits]),
        test_indices=np.stack([indices for _, indices in splits]),
        ar_coefficients=coefficients, ar_residual_variance_ratios=variance_ratios, **spectra,
    )
    (output_directory / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
