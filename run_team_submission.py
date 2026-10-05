"""Record the exact team script output and verify the whitening-off baseline."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import json
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

import whitening_compare as comparison


PROJECT_DIRECTORY = Path(__file__).resolve().parent
SUBMISSION_DIRECTORY = PROJECT_DIRECTORY / "team_submission_whole_record"


class OutputRecorder:
    def __init__(self, terminal, file_stream):
        self.terminal = terminal
        self.file_stream = file_stream

    def write(self, message):
        self.terminal.write(message)
        self.file_stream.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.file_stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whiten", choices=["off", "on"], required=True)
    arguments = parser.parse_args()
    use_whitening = arguments.whiten == "on"
    script_path = SUBMISSION_DIRECTORY / "classify_whitened.py"
    data_path = PROJECT_DIRECTORY / "source/ECoG_Handpose.mat"
    reference_path = PROJECT_DIRECTORY / "source/classify.py"
    prior_metrics = json.loads((PROJECT_DIRECTORY / "results/whitening/comparison_metrics.json").read_text())
    np.testing.assert_equal(comparison.file_sha256(data_path), prior_metrics["provenance"]["data_sha256"])
    np.testing.assert_equal(comparison.file_sha256(reference_path), prior_metrics["provenance"]["reference_script_sha256"])
    if use_whitening:
        off_metrics_path = SUBMISSION_DIRECTORY / "results/off/metrics.json"
        if not off_metrics_path.exists():
            raise ValueError("Run and verify WHITEN=False before running WHITEN=True.")
        off_metrics = json.loads(off_metrics_path.read_text())
        if not off_metrics["baseline_exactly_reproduced"] or off_metrics["script_sha256"] != comparison.file_sha256(script_path):
            raise ValueError("The whitening-off result must be verified for this exact script first.")
    output_directory = SUBMISSION_DIRECTORY / "results" / arguments.whiten
    output_directory.mkdir(parents=True, exist_ok=True)
    code = script_path.read_text()
    if code.count("WHITEN = True\n") != 1:
        raise ValueError("Expected exactly one WHITEN switch in the team script.")
    if not use_whitening:
        code = code.replace("WHITEN = True\n", "WHITEN = False\n", 1)
    figure_paths = []

    def record_figures(*positional, **named):
        for figure_number in plt.get_fignums():
            figure = plt.figure(figure_number)
            spectrum = figure.axes and figure.axes[0].get_title().startswith("Channel 26:")
            stem = "channel26_whitening_spectrum" if spectrum else "confusion_matrix"
            figure_path = output_directory / f"{stem}.png"
            figure.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
            figure.savefig(output_directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
            figure_paths.append(figure_path.name)
        plt.close("all")

    original_show = plt.show
    original_directory = Path.cwd()
    namespace = {"__name__": "__main__", "__file__": str(script_path)}
    started = time.perf_counter()
    print(f"Running the unchanged team pipeline with WHITEN={use_whitening}; only the switch is changed in memory.", flush=True)
    try:
        os.chdir(data_path.parent)
        plt.show = record_figures
        with (output_directory / "printed_output.txt").open("w") as output_stream:
            with contextlib.redirect_stdout(OutputRecorder(sys.stdout, output_stream)), threadpool_limits(limits=1):
                exec(compile(code, str(script_path), "exec"), namespace)
    finally:
        plt.show = original_show
        os.chdir(original_directory)
    baseline_matches = None
    if not use_whitening:
        with np.load(PROJECT_DIRECTORY / "results/whitening/comparison_arrays.npz") as saved:
            np.testing.assert_array_equal(namespace["X"], saved["baseline_features"])
            np.testing.assert_array_equal(namespace["acc"], saved["baseline_scores"])
            np.testing.assert_array_equal(namespace["pred"], saved["baseline_predictions"][0])
            np.testing.assert_array_equal(namespace["acc_shuf"], saved["baseline_shuffled_scores"])
        baseline = json.loads((PROJECT_DIRECTORY / "results/baseline_metrics.json").read_text())
        np.testing.assert_allclose(namespace["acc_fp"].mean() * 100, baseline["fist_peace_accuracy_mean_percent"], atol=1e-12)
        np.testing.assert_allclose(namespace["acc_fp"].std() * 100, baseline["fist_peace_accuracy_std_percent"], atol=1e-12)
        np.testing.assert_array_equal(namespace["cm"], baseline["confusion_matrix"])
        baseline_matches = True
    np.testing.assert_equal(comparison.file_sha256(data_path), prior_metrics["provenance"]["data_sha256"])
    np.testing.assert_equal(comparison.file_sha256(reference_path), prior_metrics["provenance"]["reference_script_sha256"])
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "whitening_enabled": use_whitening,
        "baseline_exactly_reproduced": baseline_matches,
        "three_class": comparison.summarize_scores(namespace["acc"]),
        "fist_peace": comparison.summarize_scores(namespace["acc_fp"]),
        "shuffled_control": comparison.summarize_scores(np.asarray(namespace["acc_shuf"])),
        "confusion_matrix_first_repeat": namespace["cm"].tolist(),
        "feature_shape": list(namespace["X"].shape), "class_names": namespace["names"],
        "filter_fit": "All 507025 continuous post-notch samples per channel" if use_whitening else None,
        "label_usage_for_filter_fit": False, "cv": {"folds": 10, "repeats": 10, "seed": 0},
        "source_script_unchanged": True, "script_sha256": comparison.file_sha256(script_path),
        "reference_sha256": comparison.file_sha256(reference_path), "data_sha256": comparison.file_sha256(data_path),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "blas_threads": 1},
        "plot_files": figure_paths, "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["One recording; no independent participant or session test.",
                       "The whole-recording filter fit includes evaluation-period signals without labels; this is the team's offline recipe.",
                       "The SD is across fold scores, not a confidence interval; repeated folds are not independent experiments.",
                       "The confusion matrix is from one complete 10-fold evaluation, not the repeated-CV mean."],
    }
    np.savez_compressed(
        output_directory / "arrays.npz", features=namespace["X"], labels=namespace["labels"],
        onsets=namespace["onsets"], three_class_scores=namespace["acc"], fist_peace_scores=namespace["acc_fp"],
        shuffled_scores=np.asarray(namespace["acc_shuf"]), first_repeat_predictions=namespace["pred"],
        confusion_matrix=namespace["cm"],
        **({"ar_coefficients": namespace["whitening_coefficients"], "spectrum_frequencies": namespace["spectrum_frequencies"],
            "spectrum_before": namespace["spectrum_before"], "spectrum_after": namespace["spectrum_after"]} if use_whitening else {}),
    )
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Recorded output and plots: {output_directory}", flush=True)
    if baseline_matches:
        print("WHITEN=False exactly reproduced the saved baseline; WHITEN=True may now be evaluated.", flush=True)


if __name__ == "__main__":
    main()
