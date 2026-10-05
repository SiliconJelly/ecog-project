"""Illustrate correct and incorrect scissors predictions with recorded sensor traces."""

from datetime import datetime, timezone
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_team_figures as figure_helpers
import whitening_compare as comparison


PROJECT_DIRECTORY = Path(__file__).resolve().parent
CLASS_NAMES = {1: "rock", 2: "scissors", 3: "paper"}
CLASS_COLORS = {1: "#4076a8", 2: "#de7839", 3: "#32966d"}


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    figure_directory = PROJECT_DIRECTORY / "results/team_figures"
    cache_path = figure_directory / "figure_data.npz"
    source_metadata = json.loads((figure_directory / "figure_metadata.json").read_text())
    current_metrics = json.loads((PROJECT_DIRECTORY / "results/whitening/comparison_metrics.json").read_text())
    if source_metadata["data_sha256"] != current_metrics["provenance"]["data_sha256"]:
        raise ValueError("The cached sensor curves and current evaluation use different recordings.")
    if comparison.file_sha256(PROJECT_DIRECTORY / "source/ECoG_Handpose.mat") != source_metadata["data_sha256"]:
        raise ValueError("The recording has changed since the sensor curves were generated.")
    with np.load(cache_path) as saved:
        cached = {name: saved[name].copy() for name in saved.files}
    with np.load(PROJECT_DIRECTORY / "results/whitening/comparison_arrays.npz") as saved:
        np.testing.assert_array_equal(cached["labels"], saved["labels"])
        np.testing.assert_array_equal(cached["onsets"], saved["onsets"])
        np.testing.assert_array_equal(cached["whitening_predictions"], saved["whitening_predictions"][0])
    times = cached["time_seconds"]
    labels = cached["labels"]
    predictions = cached["whitening_predictions"]
    gloves = cached["glove_trials"]
    brain = cached["whitening_high_gamma_db"]
    predicted_scissors = predictions == 2
    correct_indices = np.flatnonzero(predicted_scissors & (labels == 2))
    false_indices = np.flatnonzero(predicted_scissors & (labels != 2))
    if correct_indices.size == 0 or false_indices.size != 2:
        raise ValueError("This figure expects correct scissors predictions and the two observed false positives.")
    active_times = (times >= 0) & (times <= 2)
    median_glove = np.median(gloves[correct_indices], axis=0)
    distances = ((gloves[correct_indices][:, :, active_times] - median_glove[:, active_times]) ** 2).mean(axis=(1, 2))
    correct_example = int(correct_indices[np.argmin(distances)])
    ordered_false_indices = sorted(false_indices.tolist(), key=lambda trial_index: int(labels[trial_index]))
    case_indices = np.asarray([correct_example] + ordered_false_indices)
    reference_masks = {
        class_label: (labels == class_label) & (predictions == class_label)
        for class_label in [1, 2, 3]
    }
    reference_brain = {class_label: np.median(brain[mask], axis=0) for class_label, mask in reference_masks.items()}
    visible_times = times >= -0.5
    brain_values = np.concatenate([brain[case_indices][:, visible_times].ravel()] + [reference[visible_times] for reference in reference_brain.values()])
    lower_limit, upper_limit = float(brain_values.min()), float(brain_values.max())
    margin = max((upper_limit - lower_limit) * 0.08, 0.25)

    figure, axes = plt.subplots(2, 3, figsize=(12.4, 6.5), sharex=True)
    figure.suptitle("Predicted scissors: correct and incorrect trials", fontsize=12, y=0.992)
    for column_index, trial_index in enumerate(case_indices):
        cue_label = int(labels[trial_index])
        outcome = "Correct" if cue_label == 2 else "Incorrect"
        title = f"{outcome} prediction\nTrial {trial_index + 1}: predicted scissors / cue {CLASS_NAMES[cue_label]}"
        figure_helpers.plot_glove(axes[0, column_index], gloves[trial_index], title,
                                  float(cached["cue_durations_seconds"][trial_index]))
        axes[0, column_index].set_xlabel("")
        axis = axes[1, column_index]
        axis.plot(times, brain[trial_index], color="#252525", linewidth=1.3, label=f"Trial {trial_index + 1}")
        for reference_label in ([2] if cue_label == 2 else [2, cue_label]):
            reference_count = int(reference_masks[reference_label].sum())
            axis.plot(times, reference_brain[reference_label], color=CLASS_COLORS[reference_label], linewidth=1.65,
                      label=f"{CLASS_NAMES[reference_label].capitalize()} ref (n={reference_count})")
        figure_helpers.cue_axis(axis, float(cached["cue_durations_seconds"][trial_index]))
        axis.axhline(0, color="#555555", linewidth=0.65)
        axis.set_ylim(lower_limit - margin, upper_limit + margin)
        axis.set_title("High-gamma: trial vs class reference", fontsize=9.5, pad=7)
        axis.legend(frameon=False, fontsize=8, loc="best", handlelength=2)
    axes[0, 0].set_ylabel("Finger sensor (stored 0–1 scale)")
    axes[1, 0].set_ylabel("High-gamma change (dB)")
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                  ncol=5, frameon=False, fontsize=8.5, handlelength=2)
    correct_count = int(correct_indices.size)
    false_count = int(false_indices.size)
    predicted_count = int(predicted_scissors.sum())
    channel_text = ", ".join(str(channel + 1) for channel in cached["display_channel_indices"])
    figure.text(0.02, 0.065,
                f"Whitening; first complete 10-fold evaluation: {predicted_count} scissors predictions, {correct_count} matched the cue label, {false_count} did not.",
                fontsize=8)
    figure.text(0.02, 0.036,
                f"High-gamma: fixed ECoG channels {channel_text}; 50–300 Hz, 50 ms moving power, relative to -1–0 s pre-cue.", fontsize=8)
    figure.text(0.02, 0.007,
                "Shading: cue. References: medians of correctly classified trials. Cue labels describe the instruction; glove signals show execution.",
                fontsize=8)
    figure.tight_layout(rect=(0, 0.10, 1, 0.885), h_pad=1.8, w_pad=1.8)
    figure_helpers.save_figure(figure, figure_directory, "03_predicted_scissors_examples")

    np.savez_compressed(
        figure_directory / "prediction_example_data.npz", time_seconds=times,
        case_indices=case_indices, cue_labels=labels[case_indices], predictions=predictions[case_indices],
        case_glove=gloves[case_indices], case_high_gamma_db=brain[case_indices],
        reference_high_gamma_db=np.stack([reference_brain[class_label] for class_label in [1, 2, 3]]),
        reference_class_labels=np.asarray([1, 2, 3]), correct_scissors_indices=correct_indices,
        incorrect_scissors_indices=false_indices, correct_selection_distances=distances,
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": "AR(10) whitening; first complete 10-fold CV repetition.",
        "predicted_scissors_count": predicted_count,
        "correct_scissors_count": correct_count,
        "incorrect_scissors_count": false_count,
        "correct_selection": "Actual trial nearest the median glove trajectory among correct scissors predictions, by mean squared distance over 0–2 s.",
        "incorrect_selection": "Both incorrect scissors predictions included; ordered by cue class.",
        "cases": [{"trial_1_based": int(trial_index + 1), "cue": CLASS_NAMES[int(labels[trial_index])],
                   "prediction": CLASS_NAMES[int(predictions[trial_index])], "matches_cue": bool(labels[trial_index] == predictions[trial_index])}
                  for trial_index in case_indices],
        "reference_counts": {CLASS_NAMES[class_label]: int(mask.sum()) for class_label, mask in reference_masks.items()},
        "data_sha256": source_metadata["data_sha256"],
        "source_figure_data_sha256": comparison.file_sha256(cache_path),
        "script_sha256": comparison.file_sha256(Path(__file__)),
        "plot_helpers_sha256": comparison.file_sha256(PROJECT_DIRECTORY / "make_team_figures.py"),
        "label_definition": "Instruction cue labels; actual movement is observed separately using the glove.",
        "interpretation": "Examples and references from one recording; not proof of why the classifier made each decision.",
    }
    (figure_directory / "prediction_example_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
