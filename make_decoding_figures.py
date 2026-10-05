"""Explain saved rock/scissors predictions without changing the evaluated model."""

from datetime import datetime, timezone
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import nbformat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

import whitening_compare as comparison


PROJECT_DIRECTORY = Path(__file__).resolve().parent
CLASS_NAMES = {1: "Rock", 2: "Scissors", 3: "Paper"}
CLASS_COLORS = {1: "#4076a8", 2: "#de7839", 3: "#32966d"}
PATTERN_COLORS = LinearSegmentedColormap.from_list("power_difference", ["#326a9b", "#fafafa", "#c66b31"])
EXAMPLE_TRIAL_INDEX = 64


def draw_map(axis, values, limit, title):
    display = axis.imshow(
        values, origin="upper", aspect="auto", interpolation="nearest",
        extent=(0.25, 2.25, 60.5, 0.5), cmap=PATTERN_COLORS,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    axis.set_title(title, fontsize=10, pad=10)
    axis.set_xticks([0.25, 0.75, 1.25, 1.75, 2.25])
    axis.set_yticks([1, 10, 20, 30, 40, 50, 60])
    axis.set_xlabel("Time from cue (s); each column = 250 ms")
    axis.set_ylabel("ECoG electrode number")
    axis.tick_params(length=3, width=0.7)
    return display


def save_figure(figure, output_directory, stem):
    figure.savefig(output_directory / f"{stem}.png", dpi=220, facecolor="white")
    figure.savefig(output_directory / f"{stem}.svg", facecolor="white")
    plt.close(figure)


def write_viewer():
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# 가위와 바위를 모델은 어떻게 구분할까?\n\n"
            "이 파일은 **읽기 전용 결과 설명**입니다. 실행할 코드 셀이 없습니다. "
            "원본 데이터와 기존 분류·whitening 코드는 바꾸지 않았습니다.\n\n"
            "기존 그림은 전극 5개를 평균해서 차이가 가려질 수 있었습니다. "
            "여기서는 모델이 실제로 받은 **60개 전극 × 8개 시간 구간 = 480개 특징**을 펼칩니다. "
            "모든 그림은 whitening 조건입니다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 1. 평균으로 합치지 않은 전극·시간 패턴\n\n"
            "![전극과 시간별 패턴](results/team_figures/04_channel_time_patterns.png)\n\n"
            "- **가로:** 지시 후 0.25–2.25초, 한 칸은 0.25초입니다.\n"
            "- **세로:** ECoG 전극 1–60번입니다. 실제 뇌의 해부학적 배치 지도는 아닙니다.\n"
            "- **A·B:** 모델을 학습시킨 바위 27회와 가위 27회의 특징 평균입니다. "
            "각 칸의 색은 해당 특징의 학습 자료 전체 81회 평균·표준편차로 표준화한 값입니다. "
            "주황색은 학습 평균보다 높은 파워, 파란색은 낮은 파워입니다.\n"
            "- **C:** 가위 평균에서 바위 평균을 뺀 값입니다. "
            "주황색 칸은 가위가 더 높고, 파란색 칸은 바위가 더 높습니다. "
            "흰색에 가까우면 평균 차이가 작습니다. C의 색상 척도는 A·B와 별도입니다.\n\n"
            "**이 색은 동작 전 대비 dB가 아닙니다.** 전극마다 원래 파워 크기가 달라서, "
            "학습 자료 기준 표준편차 단위로 바꾸어 비교했습니다. "
            "표준화는 표시용이며 분류기에 새로 적용하지 않았습니다. "
            "오분류한 65번의 교차검증 fold를 고정해 학습 81회만으로 이 기준을 만들었습니다. "
            "테스트 9회는 기준 계산과 학습에서 제외했습니다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 2. 같은 모델의 실제 테스트 판정\n\n"
            "![실제 판정과 특징별 기여](results/team_figures/05_held_out_decisions.png)\n\n"
            "왼쪽은 **44번: 바위 → 바위**, 가운데는 **8번: 가위 → 가위**, "
            "오른쪽은 **65번: 바위 → 가위로 오분류**입니다. "
            "세 사례 모두 같은 fold의 테스트 자료이며, 이 모델은 이 세 사례로 학습하지 않았습니다.\n\n"
            "위쪽 지도는 각 특징이 **가위 점수 − 바위 점수**에 더한 값을 보여줍니다. "
            "주황색은 가위 쪽으로, 파란색은 바위 쪽으로 점수를 밀어 줍니다. "
            "480개 칸을 모두 더한 뒤 그림에 표시한 고정 offset을 더하면 실제 점수 차이가 됩니다.\n\n"
            "아래 막대는 세 동작의 모델 점수입니다. 읽기 편하게 바위 점수를 모두에서 뺀 값으로 표시했으며, "
            "이렇게 해도 가장 높은 동작은 바뀌지 않습니다. **점수는 확률이나 정확도가 아닙니다.** "
            "가장 높은 막대의 동작을 예측합니다. 오른쪽 사례는 바위 지시였지만 가위 점수가 더 높아 틀렸습니다.\n\n"
            "모델은 평균 곡선에 가장 가까운 동작을 찾는 것이 아니라, 학습한 전극·시간별 가중치로 "
            "480개 특징을 합쳐 점수를 계산합니다. 따라서 전체 곡선이 비슷해도 판정이 달라질 수 있습니다."
        ),
        nbformat.v4.new_markdown_cell(
            "## 해석 범위와 검증\n\n"
            "첫 번째 10-fold 평가의 **10개 모델·90개 테스트 예측을 같은 설정으로 재구성**했고, "
            "저장된 예측과 전부 일치했습니다. 기존 정확도를 높이기 위한 재설정이나 새 모델 선택은 하지 않았습니다.\n\n"
            "특징별 기여도는 이 LDA가 계산한 점수를 설명합니다. "
            "전극이 특정 손가락만 담당한다거나, 그 전극이 움직임의 원인이라는 뜻은 아닙니다. "
            "전극 간 상관이 있어 칸 하나의 기여를 독립적인 생물학적 중요도로 읽지 않습니다. "
            "동작 정답은 화면의 지시 라벨이며 실제 손 자세의 독립 검증값이 아닙니다.\n\n"
            "수치와 선택 기준: `results/team_figures/decoding_metadata.json`\n\n"
            "실제 지도·점수 배열: `results/team_figures/decoding_figure_data.npz`"
        ),
    ]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    nbformat.validate(notebook)
    nbformat.write(notebook, PROJECT_DIRECTORY / "04_decoding_explained.ipynb")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
    output_directory = PROJECT_DIRECTORY / "results/team_figures"
    archive_path = PROJECT_DIRECTORY / "results/whitening/comparison_arrays.npz"
    metrics_path = PROJECT_DIRECTORY / "results/whitening/comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    data_path = PROJECT_DIRECTORY / "source/ECoG_Handpose.mat"
    reference_path = PROJECT_DIRECTORY / "source/classify.py"
    np.testing.assert_equal(comparison.file_sha256(data_path), metrics["provenance"]["data_sha256"])
    np.testing.assert_equal(comparison.file_sha256(reference_path), metrics["provenance"]["reference_script_sha256"])
    with np.load(archive_path) as saved:
        features = saved["whitening_features"].copy()
        labels = saved["labels"].copy()
        saved_predictions = saved["whitening_predictions"][0].copy()
        saved_train_indices = saved["train_indices"].copy()
        saved_test_indices = saved["test_indices"].copy()
    splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0).split(features, labels))
    for split_index, (train_indices, test_indices) in enumerate(splits):
        np.testing.assert_array_equal(train_indices, saved_train_indices[split_index])
        np.testing.assert_array_equal(test_indices, saved_test_indices[split_index])
    held_out_scores = np.empty((90, 3))
    selected_classifier = None
    for fold_index, (train_indices, test_indices) in enumerate(splits[:10]):
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        with threadpool_limits(limits=1):
            classifier.fit(features[train_indices], labels[train_indices])
            fold_scores = classifier.decision_function(features[test_indices])
        np.testing.assert_array_equal(classifier.predict(features[test_indices]), saved_predictions[test_indices])
        np.testing.assert_array_equal(classifier.classes_, [1, 2, 3])
        held_out_scores[test_indices] = fold_scores
        if EXAMPLE_TRIAL_INDEX in test_indices:
            selected_classifier = classifier
            selected_fold = fold_index
            selected_train = train_indices.copy()
            selected_test = test_indices.copy()
    if selected_classifier is None:
        raise ValueError("The example is not a held-out trial in the first CV repetition.")
    train_features = features[selected_train]
    train_center = train_features.mean(axis=0)
    train_scale = train_features.std(axis=0)
    if np.any(train_scale <= 0):
        raise ValueError("A training feature has no variation.")
    class_means = np.stack([
        features[selected_train[labels[selected_train] == class_label]].mean(axis=0)
        for class_label in [1, 2]
    ])
    reference_maps = ((class_means - train_center) / train_scale).reshape(2, 60, 8)
    difference_map = ((class_means[1] - class_means[0]) / train_scale).reshape(60, 8)
    reference_limit = float(np.abs(reference_maps).max())
    difference_limit = float(np.abs(difference_map).max())

    figure, axes = plt.subplots(1, 3, figsize=(13.6, 9.0), layout="constrained")
    figure.suptitle("Scissors vs rock: the electrode-by-time patterns hidden by averaging", fontsize=12)
    reference_image = None
    for axis, values, class_label, panel in zip(axes[:2], reference_maps, [1, 2], ["A", "B"]):
        count = int(np.sum(labels[selected_train] == class_label))
        reference_image = draw_map(axis, values, reference_limit, f"{panel}  {CLASS_NAMES[class_label]} training mean (n={count})")
    difference_image = draw_map(axes[2], difference_map, difference_limit, "C  Scissors minus rock")
    figure.colorbar(reference_image, ax=axes[:2], shrink=0.72, pad=0.02,
                    label="Log power relative to training mean (training SD units)")
    figure.colorbar(difference_image, ax=axes[2], shrink=0.72, pad=0.02,
                    label="Mean difference / training SD")
    figure.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
    figure.text(0.02, 0.080, "Rows: 60 electrodes kept separate. Columns: the exact eight 250 ms log-power features used by LDA.", fontsize=9)
    figure.text(0.02, 0.051, "A-B: orange = above the training mean; blue = below. C: orange = scissors higher; blue = rock higher.", fontsize=9)
    figure.text(0.02, 0.022,
                f"Whitening; CV repeat 1, fold {selected_fold + 1}. Templates use training trials only (81 total, 27 per class). Colors are NOT pre-cue dB.", fontsize=8)
    save_figure(figure, output_directory, "04_channel_time_patterns")

    correct_rock = selected_test[(labels[selected_test] == 1) & (saved_predictions[selected_test] == 1)]
    correct_scissors = selected_test[(labels[selected_test] == 2) & (saved_predictions[selected_test] == 2)]
    if correct_rock.size == 0 or correct_scissors.size == 0:
        raise ValueError("The selected fold lacks a correct held-out example of each class.")
    case_indices = np.asarray([int(correct_rock[0]), int(correct_scissors[0]), EXAMPLE_TRIAL_INDEX])
    pair_weights = selected_classifier.coef_[1] - selected_classifier.coef_[0]
    pair_intercept = float(selected_classifier.intercept_[1] - selected_classifier.intercept_[0])
    fixed_offset = float(pair_intercept + pair_weights @ train_center)
    contributions = ((features[case_indices] - train_center) * pair_weights).reshape(3, 60, 8)
    case_scores = held_out_scores[case_indices]
    pair_margins = case_scores[:, 1] - case_scores[:, 0]
    np.testing.assert_allclose(contributions.sum(axis=(1, 2)) + fixed_offset, pair_margins, atol=1e-8, rtol=1e-10)
    relative_scores = case_scores - case_scores[:, :1]
    np.testing.assert_array_equal(np.argmax(relative_scores, axis=1) + 1, saved_predictions[case_indices])
    contribution_limit = float(np.abs(contributions).max())

    figure, axes = plt.subplots(2, 3, figsize=(13.6, 10.2), layout="constrained", gridspec_kw={"height_ratios": [3.1, 1.0]})
    figure.suptitle("What the unchanged model actually used: three held-out decisions", fontsize=12)
    contribution_image = None
    score_minimum = float(relative_scores.min())
    score_maximum = float(relative_scores.max())
    score_span = max(score_maximum - score_minimum, 1.0)
    score_limits = (score_minimum - 0.17 * score_span, score_maximum + 0.22 * score_span)
    for case_position, trial_index in enumerate(case_indices):
        cue_name = CLASS_NAMES[int(labels[trial_index])]
        prediction_name = CLASS_NAMES[int(saved_predictions[trial_index])]
        status = "correct" if labels[trial_index] == saved_predictions[trial_index] else "incorrect"
        title = f"Trial {trial_index + 1}: cue {cue_name} / predicted {prediction_name}\n{status.capitalize()}; feature contributions to scissors minus rock"
        contribution_image = draw_map(axes[0, case_position], contributions[case_position], contribution_limit, title)
        axis = axes[1, case_position]
        axis.barh(np.arange(3), relative_scores[case_position], color=[CLASS_COLORS[class_label] for class_label in [1, 2, 3]], height=0.58)
        axis.set_yticks(np.arange(3), [CLASS_NAMES[class_label] for class_label in [1, 2, 3]])
        axis.invert_yaxis()
        axis.axvline(0, color="#333333", linewidth=0.7)
        axis.set_xlim(score_limits)
        axis.set_xlabel("LDA score minus rock score (not a probability)")
        axis.spines[["top", "right"]].set_visible(False)
        for class_position, value in enumerate(relative_scores[case_position]):
            axis.text(value + 0.014 * score_span, class_position, f"{value:+.1f}", va="center", fontsize=8)
        axis.set_title(f"Highest score wins: {prediction_name}", color=CLASS_COLORS[int(saved_predictions[trial_index])], fontsize=10, pad=8)
    figure.colorbar(contribution_image, ax=axes[0, :], shrink=0.85, pad=0.02,
                    label="Contribution to scissors minus rock score (orange: scissors; blue: rock)")
    figure.get_layout_engine().set(rect=(0, 0.125, 1, 0.875))
    figure.text(0.02, 0.087,
                f"Exact decomposition: sum of 480 colored cells + fixed offset ({fixed_offset:+.2f}) = scissors-minus-rock score.", fontsize=9)
    margin_text = "; ".join(f"trial {trial_index + 1}: {margin:+.2f}" for trial_index, margin in zip(case_indices, pair_margins))
    figure.text(0.02, 0.059, f"Score differences: {margin_text}. Negative favors rock; positive favors scissors (paper must also be compared).", fontsize=8)
    figure.text(0.02, 0.031,
                f"All three trials were held out from the SAME model (repeat 1, fold {selected_fold + 1}; training 81, testing 9). No new feature selection or tuning.", fontsize=8)
    figure.text(0.02, 0.008, "This explains the fitted LDA calculation, not the biological cause of a movement or an error. Labels are instruction cues.", fontsize=8)
    save_figure(figure, output_directory, "05_held_out_decisions")

    all_relative_scores = held_out_scores - held_out_scores[:, :1]
    np.testing.assert_array_equal(np.argmax(all_relative_scores, axis=1) + 1, saved_predictions)
    np.savez_compressed(
        output_directory / "decoding_figure_data.npz", reference_maps=reference_maps,
        difference_map=difference_map, train_indices=selected_train, test_indices=selected_test,
        train_center=train_center, train_scale=train_scale, class_means=class_means,
        case_indices=case_indices, cue_labels=labels[case_indices], predictions=saved_predictions[case_indices],
        contributions=contributions, pair_weights=pair_weights, pair_intercept=pair_intercept,
        fixed_offset=fixed_offset, case_scores=case_scores, relative_scores=relative_scores,
        pair_margins=pair_margins, all_held_out_scores=held_out_scores,
        model_coefficients=selected_classifier.coef_, model_intercepts=selected_classifier.intercept_,
        bin_edges_seconds=comparison.BIN_EDGES_SECONDS,
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": "Existing AR(10) whitening features; unchanged shrinkage LDA.",
        "reference_cv_repeat_1_based": 1, "reference_cv_fold_1_based": selected_fold + 1,
        "training_count": int(selected_train.size), "test_count": int(selected_test.size),
        "training_class_counts": {CLASS_NAMES[class_label]: int(np.sum(labels[selected_train] == class_label)) for class_label in [1, 2, 3]},
        "template_definition": "Class means from all relevant training trials, not only correctly predicted trials; centered/scaled per feature by all 81 training trials for display only.",
        "decomposition": "(coef_scissors - coef_rock) * (test_features - training_feature_mean); fixed offset includes the original intercept and centering correction.",
        "fixed_offset": fixed_offset,
        "cases": [{"trial_1_based": int(trial_index + 1), "cue": CLASS_NAMES[int(labels[trial_index])],
                   "prediction": CLASS_NAMES[int(saved_predictions[trial_index])],
                   "scores_relative_to_rock": relative_scores[case_position].tolist(),
                   "feature_contribution_sum": float(contributions[case_position].sum()),
                   "scissors_minus_rock_score": float(pair_margins[case_position])}
                  for case_position, trial_index in enumerate(case_indices)],
        "case_selection": "Trial 65 is the previously shown rock-to-scissors error; first chronological correct rock and scissors among the same fold's test trials.",
        "validation": {"first_repeat_saved_predictions_reproduced": True, "held_out_trial_count": 90,
                       "all_100_saved_splits_match": True, "no_training_test_overlap": bool(np.intersect1d(selected_train, selected_test).size == 0),
                       "maximum_score_decomposition_error": float(np.abs(contributions.sum(axis=(1, 2)) + fixed_offset - pair_margins).max())},
        "data_sha256": comparison.file_sha256(data_path),
        "reference_script_sha256": comparison.file_sha256(reference_path),
        "feature_archive_sha256": comparison.file_sha256(archive_path),
        "metrics_sha256": comparison.file_sha256(metrics_path),
        "script_sha256": comparison.file_sha256(Path(__file__)),
        "limits": ["One recording; cue labels are instructions, not independently verified hand posture.",
                   "A channel number is not an anatomical location without an electrode map.",
                   "LDA contributions describe a fitted calculation, not causal biological importance; correlated features share information.",
                   "Display scaling is not additional model preprocessing; templates use training trials only."],
    }
    (output_directory / "decoding_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_viewer()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
