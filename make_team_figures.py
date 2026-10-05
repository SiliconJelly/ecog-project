"""Create shareable figures from recorded glove signals and verified CV predictions."""

from datetime import datetime, timezone
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits

import whitening_compare as comparison


PROJECT_DIRECTORY = Path(__file__).resolve().parent
SAMPLING_RATE = 1200
SAMPLE_STEP = 6
EPOCH_OFFSETS = np.arange(-SAMPLING_RATE, 3 * SAMPLING_RATE, SAMPLE_STEP)
TIME_SECONDS = EPOCH_OFFSETS / SAMPLING_RATE
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_COLORS = ["#3978c6", "#e47736", "#44a77f", "#d8a12a", "#cf709d"]
CLASS_NAMES = {1: "Fist", 2: "Peace", 3: "Open"}
REFERENCE_COLORS = {3: "#32966d", 2: "#de7839"}


def epoch_signal(record, onsets):
    indices = onsets[:, None] + EPOCH_OFFSETS[None, :]
    if indices.min() < 0 or indices.max() >= record.shape[1]:
        raise ValueError("A visualization epoch extends beyond the recording.")
    return np.stack([record[:, trial_indices] for trial_indices in indices])


def high_gamma_trials(cleaned, onsets):
    bandpass = signal.butter(4, [50, 300], "bandpass", fs=SAMPLING_RATE, output="sos")
    power = signal.sosfiltfilt(bandpass, cleaned, axis=1)
    np.square(power, out=power)
    uniform_filter1d(power, size=60, axis=1, mode="constant", output=power)
    np.maximum(power, np.finfo(np.float64).tiny, out=power)
    np.log10(power, out=power)
    power *= 10
    trials = epoch_signal(power, onsets)
    trials -= trials[:, :, TIME_SECONDS < 0].mean(axis=2, keepdims=True)
    if not np.isfinite(trials).all():
        raise ValueError("High-gamma time courses contain non-finite values.")
    return trials


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
    axis.tick_params(width=0.7, length=3)


def cue_axis(axis, cue_duration):
    axis.axvspan(0, cue_duration, color="#e9e9e9", alpha=0.65, linewidth=0)
    axis.set_xlim(-0.5, 3.0)
    axis.set_xticks(np.arange(-0.5, 3.1, 0.5))
    axis.set_xlabel("Time from cue (s)")
    style_axis(axis)


def plot_glove(axis, trajectories, title, cue_duration, legend=False):
    for finger_index, (finger_name, color) in enumerate(zip(FINGER_NAMES, FINGER_COLORS)):
        axis.plot(TIME_SECONDS, trajectories[finger_index], color=color, linewidth=1.45, label=finger_name)
    cue_axis(axis, cue_duration)
    axis.set_ylim(-0.04, 1.05)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_title(title, fontsize=9.5, pad=8)
    if legend:
        axis.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.8)


def plot_brain(axis, brain_trials, reference_masks, example_indices, cue_duration):
    for class_label in [3, 2]:
        reference = np.median(brain_trials[reference_masks[class_label]], axis=0)
        axis.plot(TIME_SECONDS, reference, color=REFERENCE_COLORS[class_label], linewidth=1.8,
                  label=f"{CLASS_NAMES[class_label]} reference")
    for example_index, line_style in zip(example_indices, ["-", "--"]):
        axis.plot(TIME_SECONDS, brain_trials[example_index], color="#292929", linewidth=1.05,
                  linestyle=line_style, label=f"Trial {example_index + 1}")
    cue_axis(axis, cue_duration)
    axis.axhline(0, color="#555555", linewidth=0.65)
    axis.set_ylabel("High-gamma change (dB)")
    axis.legend(frameon=False, fontsize=7.5, loc="best", handlelength=2.1)


def save_figure(figure, output_directory, stem):
    figure.savefig(output_directory / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output_directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    result_directory = PROJECT_DIRECTORY / "results/whitening"
    metrics = json.loads((result_directory / "comparison_metrics.json").read_text())
    data_path = PROJECT_DIRECTORY / "source/ECoG_Handpose.mat"
    data_hash = comparison.file_sha256(data_path)
    if data_hash != metrics["provenance"]["data_sha256"]:
        raise ValueError("The recording differs from the evaluated data.")
    if comparison.file_sha256(PROJECT_DIRECTORY / "whitening_compare.py") != metrics["provenance"]["analysis_script_sha256"]:
        raise ValueError("The comparison implementation has changed since evaluation.")
    with np.load(result_directory / "comparison_arrays.npz") as saved:
        labels = saved["labels"].copy()
        onsets = saved["onsets"].copy()
        baseline_predictions = saved["baseline_predictions"][0].copy()
        whitening_predictions = saved["whitening_predictions"][0].copy()
        coefficients = saved["ar_coefficients"].copy()
        baseline_scores = saved["baseline_scores"].copy()
        whitening_scores = saved["whitening_scores"].copy()
    example_indices = np.flatnonzero((labels == 3) & (baseline_predictions != labels))
    if example_indices.size != 2:
        raise ValueError("The figure expects the two recorded baseline Open errors.")
    reference_masks = {
        class_label: (labels == class_label) & (baseline_predictions == labels) & (whitening_predictions == labels)
        for class_label in [3, 2]
    }
    if any(mask.sum() == 0 for mask in reference_masks.values()):
        raise ValueError("No correctly classified reference trials are available.")
    print("1/3 Reading the actual glove signals and cue durations.", flush=True)
    record = loadmat(data_path, variable_names=["y"])["y"]
    cue = record[61].copy()
    np.testing.assert_array_equal(cue[onsets].astype(int), labels)
    detected_onsets = np.flatnonzero((np.diff(cue) != 0) & (cue[1:] != 0)) + 1
    np.testing.assert_array_equal(detected_onsets, onsets)
    glove_trials = epoch_signal(record[62:67], onsets)
    raw_signal = record[1:61].copy()
    del record
    cue_durations = np.asarray([
        np.flatnonzero(cue[onset:] != cue[onset])[0] / SAMPLING_RATE for onset in onsets
    ])
    print("2/3 Computing high-gamma time courses with the evaluated preprocessing.", flush=True)
    cleaned = comparison.clean_signal(raw_signal)
    del raw_signal
    baseline_trials = high_gamma_trials(cleaned, onsets)
    whitened = comparison.apply_whitening(cleaned, coefficients)
    del cleaned
    whitening_trials = high_gamma_trials(whitened, onsets)
    del whitened
    active_times = (TIME_SECONDS >= 0.5) & (TIME_SECONDS <= 2.5)
    channel_activation = baseline_trials[:, :, active_times].mean(axis=(0, 2))
    display_channels = np.argsort(channel_activation)[::-1][:5]
    baseline_brain = baseline_trials[:, display_channels].mean(axis=1)
    whitening_brain = whitening_trials[:, display_channels].mean(axis=1)
    del baseline_trials, whitening_trials
    output_directory = PROJECT_DIRECTORY / "results/team_figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    channel_text = ", ".join(str(channel + 1) for channel in display_channels)
    cue_duration = float(np.median(cue_durations))
    print("3/3 Drawing the error examples and the paired whitening comparison.", flush=True)
    figure, axes = plt.subplots(1, 5, figsize=(18, 4.0))
    for axis, class_label in zip(axes[:2], [3, 2]):
        mask = reference_masks[class_label]
        plot_glove(axis, np.median(glove_trials[mask], axis=0),
                   f"{CLASS_NAMES[class_label]} reference (n={int(mask.sum())})",
                   float(np.median(cue_durations[mask])), legend=class_label == 3)
    axes[0].set_ylabel("Finger sensor (stored 0–1 scale)")
    for axis, example_index in zip(axes[2:4], example_indices):
        plot_glove(axis, glove_trials[example_index],
                   f"Trial {example_index + 1}: cue Open\nBaseline: {CLASS_NAMES[int(baseline_predictions[example_index])]} / Whitening: {CLASS_NAMES[int(whitening_predictions[example_index])]}",
                   float(cue_durations[example_index]))
    plot_brain(axes[4], baseline_brain, reference_masks, example_indices, cue_duration)
    axes[4].set_title("ECoG high-gamma: baseline", fontsize=9.5, pad=8)
    figure.text(0.02, 0.045,
                "Shading: cue. References: pointwise medians of trials correct under both methods. Errors: first CV repetition.",
                fontsize=8)
    figure.text(0.02, 0.008,
                f"High-gamma: 50–300 Hz, 50 ms moving power, relative to -1–0 s pre-cue; mean of ECoG channels {channel_text}.",
                fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.8)
    save_figure(figure, output_directory, "01_glove_ecog_errors")

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]})
    brain_values = np.concatenate([
        brain[example_indices].ravel() for brain in [baseline_brain, whitening_brain]
    ] + [
        np.median(brain[mask], axis=0) for brain in [baseline_brain, whitening_brain]
        for mask in reference_masks.values()
    ])
    lower_limit, upper_limit = float(brain_values.min()), float(brain_values.max())
    margin = max((upper_limit - lower_limit) * 0.08, 0.25)
    for axis, brain, title in zip(axes[:2], [baseline_brain, whitening_brain], ["A  Before whitening", "B  After whitening"]):
        plot_brain(axis, brain, reference_masks, example_indices, cue_duration)
        axis.set_ylim(lower_limit - margin, upper_limit + margin)
        axis.set_title(title, fontsize=10, loc="left", pad=9)
    baseline_repeat = baseline_scores.reshape(10, 10).mean(axis=1) * 100
    whitening_repeat = whitening_scores.reshape(10, 10).mean(axis=1) * 100
    np.testing.assert_allclose(baseline_repeat, metrics["per_repeat_accuracy_percent"]["baseline"])
    np.testing.assert_allclose(whitening_repeat, metrics["per_repeat_accuracy_percent"]["whitening"])
    axis = axes[2]
    for repeat_index, horizontal_offset in enumerate(np.linspace(-0.065, 0.065, 10)):
        positions = np.array([0.0, 1.0]) + horizontal_offset
        accuracies = [baseline_repeat[repeat_index], whitening_repeat[repeat_index]]
        axis.plot(positions, accuracies, color="#c5c5c5", linewidth=0.85, zorder=1)
        axis.scatter(positions, accuracies, c=["#487c9a", "#bc6743"], s=24, zorder=2)
    means = [baseline_repeat.mean(), whitening_repeat.mean()]
    axis.scatter([0, 1], means, color="#222222", marker="D", s=38, zorder=3)
    axis.set_xticks([0, 1], [f"Baseline\nMean {means[0]:.1f}%", f"Whitening\nMean {means[1]:.1f}%"])
    axis.set(xlim=(-0.35, 1.35), ylim=(90, 100), ylabel="Three-class accuracy (%)")
    axis.set_title(f"C  Paired CV repeats (+{metrics['paired_difference_pp']:.2f} pp)", fontsize=10, loc="left", pad=9)
    style_axis(axis)
    figure.text(0.02, 0.06,
                f"A–B: same five channels ({channel_text}), references and trials; 50 ms power used for visualization only.", fontsize=8)
    figure.text(0.02, 0.015,
                "C: one point per complete 10-fold repetition, same 90 trials; diamonds = means. Trial 22 corrected; trial 1 remains an error.", fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.0)
    save_figure(figure, output_directory, "02_whitening_comparison")

    np.savez_compressed(
        output_directory / "figure_data.npz", time_seconds=TIME_SECONDS, glove_trials=glove_trials,
        baseline_high_gamma_db=baseline_brain, whitening_high_gamma_db=whitening_brain,
        display_channel_indices=display_channels, channel_activation_db=channel_activation,
        example_trial_indices=example_indices, open_reference_mask=reference_masks[3],
        peace_reference_mask=reference_masks[2], baseline_repeat_accuracy=baseline_repeat,
        whitening_repeat_accuracy=whitening_repeat, cue_durations_seconds=cue_durations,
        labels=labels, onsets=onsets, baseline_predictions=baseline_predictions,
        whitening_predictions=whitening_predictions,
    )
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": data_hash,
        "prediction_arrays_sha256": comparison.file_sha256(result_directory / "comparison_arrays.npz"),
        "comparison_metrics_sha256": comparison.file_sha256(result_directory / "comparison_metrics.json"),
        "figure_script_sha256": comparison.file_sha256(Path(__file__)),
        "ecog_channels_1_based": (display_channels + 1).tolist(),
        "channel_selection": "Five largest pooled baseline high-gamma changes, 0.5–2.5 s over all trials; visualization only, fixed for both conditions.",
        "glove_scale": "Stored sensor values; no additional scaling or calibration.",
        "reference_definition": "Pointwise median over trials correct under both conditions in the first CV repetition.",
        "reference_counts": {CLASS_NAMES[class_label]: int(mask.sum()) for class_label, mask in reference_masks.items()},
        "label_definition": "Class labels come from the instruction cue; glove signals record actual execution and may differ from the instruction.",
        "examples": [{"trial_1_based": int(example_index + 1), "cue_class": CLASS_NAMES[int(labels[example_index])],
                      "baseline_prediction": CLASS_NAMES[int(baseline_predictions[example_index])],
                      "whitening_prediction": CLASS_NAMES[int(whitening_predictions[example_index])]}
                     for example_index in example_indices],
        "example_selection": "All baseline Open errors from the first CV repetition; no cases excluded.",
        "high_gamma": {"band_hz": [50, 300], "power_smoothing_ms": 50,
                       "baseline_seconds": [-1, 0], "baseline_method": "Subtract each channel/trial's mean log power over pre-cue samples.",
                       "channel_aggregation": "Arithmetic mean of baseline-corrected dB over the fixed five channels.",
                       "display_sample_interval_ms": 5},
        "accuracy_gain_percentage_points": metrics["paired_difference_pp"],
        "limitations": ["Exploratory illustration from one recording.",
                        "Reference traces and display channels are selected from the same recording, not independent test data.",
                        "Time-course visualization does not establish the cause of a classification error.",
                        "The classifier uses the unchanged 250 ms features; the plotted 50 ms power is visualization only."],
    }
    (output_directory / "figure_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
