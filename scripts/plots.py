"""使用 OpenCV 绘制训练曲线、逐类 PR/ROC 曲线和混淆矩阵。"""

from pathlib import Path
from typing import Dict, List, Tuple
import json

import cv2
import numpy as np

import config


def update_probability_histogram(
    histogram: Tuple[np.ndarray, np.ndarray],
    probabilities: np.ndarray,
    masks: np.ndarray,
) -> None:
    """按类别累计像素预测分数直方图，避免保存整个测试集概率张量。"""
    positive_counts, negative_counts = histogram
    bins = positive_counts.shape[1]
    probabilities = np.clip(probabilities, 0.0, 1.0)

    for class_id in range(positive_counts.shape[0]):
        scores = probabilities[:, class_id].reshape(-1)
        targets = masks.reshape(-1) == class_id
        score_bins = np.minimum((scores * bins).astype(np.int32), bins - 1)
        positive_counts[class_id] += np.bincount(
            score_bins[targets], minlength=bins
        )
        negative_counts[class_id] += np.bincount(
            score_bins[~targets], minlength=bins
        )


def _curve_data(histogram: Tuple[np.ndarray, np.ndarray]) -> List[Dict[str, object]]:
    """从分数直方图计算每个类别的 PR、ROC、F1 曲线和面积指标。"""
    positive_counts, negative_counts = histogram
    bins = positive_counts.shape[1]
    curves = []

    for class_id, (positive, negative) in enumerate(zip(positive_counts, negative_counts)):
        true_positive = np.concatenate(([0], np.cumsum(positive[::-1], dtype=np.int64)))
        false_positive = np.concatenate(([0], np.cumsum(negative[::-1], dtype=np.int64)))
        total_positive = max(int(positive.sum()), 1)
        total_negative = max(int(negative.sum()), 1)
        recall = true_positive / total_positive
        precision = true_positive / np.maximum(true_positive + false_positive, 1)
        precision[0] = 1.0
        false_positive_rate = false_positive / total_negative
        true_positive_rate = recall.copy()
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
        thresholds = np.concatenate(([1.0 + 1.0 / bins], np.arange(bins - 1, -1, -1) / bins))

        average_precision = float(np.sum(np.diff(recall) * precision[1:]))
        roc_auc = float(
            np.sum(
                np.diff(false_positive_rate)
                * (true_positive_rate[:-1] + true_positive_rate[1:])
                * 0.5
            )
        )
        curves.append(
            {
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "f1": f1,
                "thresholds": thresholds,
                "average_precision": average_precision,
                "roc_auc": roc_auc,
            }
        )
    return curves


def _class_label(class_id: int, class_names=None) -> str:
    names = tuple(class_names) if class_names is not None else config.PLOT_CLASS_NAMES
    if class_id < len(names):
        return names[class_id]
    return f"Class {class_id}"


def _draw_chart(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    series: List[Dict[str, object]],
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    highlight_x: float = None,
    highlight_y: float = None,
    highlight_label: str = "BEST",
) -> None:
    """在指定画布区域绘制折线图和图例。"""
    x, y, width, height = rect
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width - 1, y + height - 1),
        config.PLOT_BACKGROUND_BGR,
        thickness=-1,
    )
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width - 1, y + height - 1),
        config.PLOT_GRID_BGR,
        thickness=1,
    )
    cv2.putText(
        canvas, title, (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, config.PLOT_AXIS_BGR, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, y_label, (x + 20, y + 51), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
    )

    left, top, right, bottom = 72, 62, 20, 72
    plot_left = x + left
    plot_top = y + top
    plot_width = max(width - left - right, 1)
    plot_height = max(height - top - bottom, 1)
    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    for tick in range(5):
        ratio = tick / 4.0
        grid_y = int(round(plot_top + plot_height * (1.0 - ratio)))
        grid_x = int(round(plot_left + plot_width * ratio))
        cv2.line(
            canvas, (plot_left, grid_y), (plot_left + plot_width, grid_y),
            config.PLOT_GRID_BGR, 1, cv2.LINE_AA,
        )
        cv2.line(
            canvas, (grid_x, plot_top), (grid_x, plot_top + plot_height),
            config.PLOT_GRID_BGR, 1, cv2.LINE_AA,
        )
        y_value = y_min + ratio * (y_max - y_min)
        x_value = x_min + ratio * (x_max - x_min)
        y_text = f"{y_value:.2g}"
        x_text = f"{int(round(x_value))}" if x_label.lower() == "epoch" else f"{x_value:.2g}"
        cv2.putText(
            canvas, y_text, (x + 5, grid_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, x_text, (grid_x - 16, plot_top + plot_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
        )

    cv2.line(
        canvas, (plot_left, plot_top), (plot_left, plot_top + plot_height),
        config.PLOT_AXIS_BGR, 2, cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (plot_left, plot_top + plot_height),
        (plot_left + plot_width, plot_top + plot_height),
        config.PLOT_AXIS_BGR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas, x_label, (plot_left + max(plot_width // 2 - 35, 0), y + height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
    )

    for item in series:
        values = np.asarray(item["y"], dtype=np.float64)
        if values.size == 0:
            continue
        x_values = item.get("x")
        if x_values is None:
            x_values = np.linspace(x_min, x_max, values.size)
        else:
            x_values = np.asarray(x_values, dtype=np.float64)
        if x_values.size != values.size:
            continue
        points = []
        for x_value, y_value in zip(x_values, values):
            px = int(round(plot_left + (x_value - x_min) / (x_max - x_min) * plot_width))
            py = int(round(plot_top + (y_max - y_value) / (y_max - y_min) * plot_height))
            points.append((px, py))
        if len(points) > 1:
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=item["color"],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    if highlight_x is not None and x_min <= highlight_x <= x_max:
        marker_x = int(round(plot_left + (highlight_x - x_min) / (x_max - x_min) * plot_width))
        dash_height = 12
        for dash_top in range(plot_top, plot_top + plot_height, dash_height * 2):
            cv2.line(
                canvas,
                (marker_x, dash_top),
                (marker_x, min(dash_top + dash_height, plot_top + plot_height)),
                config.BEST_MARKER_BGR,
                1,
                cv2.LINE_AA,
            )
        if highlight_y is not None and y_min <= highlight_y <= y_max:
            marker_y = int(round(plot_top + (y_max - highlight_y) / (y_max - y_min) * plot_height))
            cv2.circle(canvas, (marker_x, marker_y), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (marker_x, marker_y), 6, config.BEST_MARKER_BGR, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                highlight_label,
                (min(marker_x + 9, plot_left + plot_width - 68), max(marker_y - 8, plot_top + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                config.BEST_MARKER_BGR,
                1,
                cv2.LINE_AA,
            )

    legend_y = y + height - 38
    cursor_x = x + 18
    for item in series:
        label = str(item["label"])
        cv2.line(
            canvas, (cursor_x, legend_y), (cursor_x + 24, legend_y),
            item["color"], 3, cv2.LINE_AA,
        )
        text_x = cursor_x + 30
        cv2.putText(
            canvas, label, (text_x, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.40, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
        )
        cursor_x = text_x + max(len(label) * 8, 70)


def save_training_results_plot(
    history: List[Dict[str, object]],
    output_path: Path,
    best_epoch: int = None,
    final_metrics: Dict[str, object] = None,
    validation_source: str = "validation",
) -> None:
    """生成含最佳轮次标记和关键指标摘要的六宫格训练总览图。"""
    width = config.RESULTS_PLOT_WIDTH
    height = config.RESULTS_PLOT_HEIGHT
    canvas = np.full((height, width, 3), config.PLOT_BACKGROUND_BGR, dtype=np.uint8)
    if not history:
        cv2.putText(
            canvas, "No training history found", (60, 100), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, config.PLOT_AXIS_BGR, 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), canvas)
        return

    best_record = None
    if best_epoch is not None:
        best_record = next((item for item in history if item["epoch"] == best_epoch), None)
    if best_record is None:
        best_record = max(history, key=lambda item: item["val"]["mIoU"])
    best_epoch = int(best_record["epoch"])
    best_metrics = best_record["val"]

    cv2.putText(
        canvas,
        "Training Results | Orange marker = selected checkpoint",
        (42, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        config.PLOT_AXIS_BGR,
        2,
        cv2.LINE_AA,
    )
    best_summary = (
        f"Best epoch {best_epoch} | Val mIoU {best_metrics['mIoU']:.4f} | "
        f"mDice {best_metrics['mDice']:.4f} | Acc {best_metrics['accuracy']:.4f} | "
        f"Loss {best_metrics['loss']:.4f} | LR {float(best_record['lr']):.2e}"
    )
    cv2.putText(
        canvas,
        best_summary,
        (42, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        config.PLOT_AXIS_BGR,
        1,
        cv2.LINE_AA,
    )
    final_summary = f"Validation source: {validation_source}"
    if final_metrics:
        final_summary += (
            f" | Final eval mIoU {final_metrics['mIoU']:.4f}"
            f" | mDice {final_metrics['mDice']:.4f}"
            f" | Acc {final_metrics['accuracy']:.4f}"
        )
    cv2.putText(
        canvas,
        final_summary,
        (42, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        config.PLOT_AXIS_BGR,
        1,
        cv2.LINE_AA,
    )

    epochs = np.asarray([item["epoch"] for item in history], dtype=np.float64)
    panels = [
        ("Total Loss", "Loss", [("Train", [item["train"]["loss"] for item in history], config.PLOT_COLORS_BGR[0]),
                                 ("Validation", [item["val"]["loss"] for item in history], config.PLOT_COLORS_BGR[2])], None),
        ("mIoU", "Score", [("Train", [item["train"]["mIoU"] for item in history], config.PLOT_COLORS_BGR[0]),
                            ("Validation", [item["val"]["mIoU"] for item in history], config.PLOT_COLORS_BGR[2])], (0.0, 1.0)),
        ("mDice", "Score", [("Train", [item["train"]["mDice"] for item in history], config.PLOT_COLORS_BGR[0]),
                             ("Validation", [item["val"]["mDice"] for item in history], config.PLOT_COLORS_BGR[2])], (0.0, 1.0)),
        ("Pixel Accuracy", "Score", [("Train", [item["train"]["accuracy"] for item in history], config.PLOT_COLORS_BGR[0]),
                                      ("Validation", [item["val"]["accuracy"] for item in history], config.PLOT_COLORS_BGR[2])], (0.0, 1.0)),
        ("Main Loss", "Loss", [("Train", [item["train"]["main_loss"] for item in history], config.PLOT_COLORS_BGR[0]),
                                ("Validation", [item["val"]["main_loss"] for item in history], config.PLOT_COLORS_BGR[2])], None),
        ("Learning Rate", "Rate", [("Learning rate", [item["lr"] for item in history], config.PLOT_COLORS_BGR[3])], None),
    ]
    panel_width = width // 3
    panel_height = (height - config.RESULTS_HEADER_HEIGHT) // 2
    x_range = (float(epochs[0]), float(epochs[-1]))
    if x_range[0] == x_range[1]:
        x_range = (x_range[0], x_range[0] + 1.0)

    for index, (title, y_label, raw_series, y_range) in enumerate(panels):
        series = [
            {"label": label, "y": np.asarray(values), "color": color, "x": epochs}
            for label, values, color in raw_series
        ]
        if y_range is None:
            values = np.concatenate([item["y"] for item in series])
            upper = max(float(np.max(values)) * 1.1, 1e-6)
            y_range = (min(0.0, float(np.min(values))), upper)
        col = index % 3
        row = index // 3
        rect = (
            col * panel_width,
            config.RESULTS_HEADER_HEIGHT + row * panel_height,
            panel_width,
            panel_height,
        )
        target_series = None
        if title == "Learning Rate":
            target_series = series[0]
        else:
            target_series = next((item for item in series if item["label"] == "Validation"), None)
        best_index = int(np.argmin(np.abs(epochs - best_epoch)))
        highlight_y = (
            float(np.asarray(target_series["y"])[best_index])
            if target_series is not None
            else None
        )
        _draw_chart(
            canvas, rect, title, "Epoch", y_label, series,
            x_range=x_range, y_range=y_range,
            highlight_x=float(best_epoch),
            highlight_y=highlight_y,
            highlight_label="BEST" if title == "mIoU" else "",
        )
    cv2.imwrite(str(output_path), canvas)


def save_training_history_jsonl(history: List[Dict[str, object]], output_path: Path) -> None:
    """保存完整机器可读训练历史，供后续重新绘图和评估使用。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as history_file:
        for item in history:
            history_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)


def save_training_history_text(
    history: List[Dict[str, object]],
    output_path: Path,
    model_name: str,
    validation_source: str,
    best_epoch: int = None,
    final_metrics: Dict[str, object] = None,
    training_seconds: float = None,
    class_names: Tuple[str, ...] = None,
) -> None:
    """将训练摘要与逐轮指标保存为适合直接阅读的文本表格。"""
    if not history:
        return

    best_record = None
    if best_epoch is not None:
        best_record = next((item for item in history if item["epoch"] == best_epoch), None)
    if best_record is None:
        best_record = max(history, key=lambda item: item["val"]["mIoU"])
    best_metrics = best_record["val"]
    lines = [
        f"{model_name} 训练报告",
        "=" * 116,
        f"验证集来源：{validation_source}",
        f"完成轮数：{max(int(item['epoch']) for item in history)}",
        f"最佳权重：第 {int(best_record['epoch'])} 轮（按验证集 mIoU 选择）",
        (
            f"最佳轮次验证指标：mIoU={best_metrics['mIoU']:.4f} | "
            f"mDice={best_metrics['mDice']:.4f} | 像素准确率={best_metrics['accuracy']:.4f} | "
            f"Loss={best_metrics['loss']:.4f} | LR={float(best_record['lr']):.2e}"
        ),
    ]
    if training_seconds is not None:
        lines.append(f"训练耗时：{training_seconds / 60.0:.1f} 分钟")
    if final_metrics:
        lines.append(
            f"最终评估指标：mIoU={final_metrics['mIoU']:.4f} | "
            f"mDice={final_metrics['mDice']:.4f} | 像素准确率={final_metrics['accuracy']:.4f} | "
            f"Loss={final_metrics['loss']:.4f}"
        )
        class_names = tuple(class_names or config.CLASS_NAMES)
        lines.append("各类别最终 IoU / Dice：")
        for index, iou in enumerate(final_metrics.get("per_class_iou", [])):
            dice = final_metrics.get("per_class_dice", [])[index]
            name = class_names[index] if index < len(class_names) else f"Class {index}"
            lines.append(f"  {name:<8} IoU={iou:.4f}  Dice={dice:.4f}")

    lines.extend([
        "",
        "训练过程（* 表示最佳权重所在轮次）",
        "*  Epoch       LR   TrainLoss     ValLoss   TrainIoU     ValIoU   ValDice    ValAcc",
        "-" * 88,
    ])
    for item in history:
        train_metrics = item["train"]
        val_metrics = item["val"]
        marker = "*" if int(item["epoch"]) == int(best_record["epoch"]) else " "
        lines.append(
            f"{marker}  {int(item['epoch']):5d}  {float(item['lr']):.2e}  "
            f"{train_metrics['loss']:10.4f}  {val_metrics['loss']:10.4f}  "
            f"{train_metrics['mIoU']:10.4f}  {val_metrics['mIoU']:10.4f}  "
            f"{val_metrics['mDice']:8.4f}  {val_metrics['accuracy']:8.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    temporary_path.replace(output_path)


def save_evaluation_plots(
    histogram: Tuple[np.ndarray, np.ndarray],
    confusion_matrix: List[List[int]],
    class_names: Tuple[str, ...],
    output_dir: Path,
    plot_class_names: Tuple[str, ...] = None,
) -> Dict[str, Dict[str, float]]:
    """保存测试集曲线和混淆矩阵，并返回每类曲线面积指标。"""
    curves = _curve_data(histogram)
    display_names = tuple(plot_class_names or class_names)
    curve_metrics: Dict[str, Dict[str, float]] = {}
    colors = config.PLOT_COLORS_BGR
    precision_series = []
    roc_series = [{
        "label": "Random baseline",
        "x": np.asarray([0.0, 1.0]),
        "y": np.asarray([0.0, 1.0]),
        "color": config.PLOT_GRID_BGR,
    }]
    f1_series = []

    for curve in curves:
        class_id = int(curve["class_id"])
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        color = colors[class_id % len(colors)]
        average_precision = float(curve["average_precision"])
        roc_auc = float(curve["roc_auc"])
        curve_metrics[class_name] = {
            "average_precision": average_precision,
            "roc_auc": roc_auc,
        }
        precision_series.append(
            {
                "label": f"{_class_label(class_id, display_names)} AP={average_precision:.3f}",
                "x": curve["recall"],
                "y": curve["precision"],
                "color": color,
            }
        )
        roc_series.append(
            {
                "label": f"{_class_label(class_id, display_names)} AUC={roc_auc:.3f}",
                "x": curve["false_positive_rate"],
                "y": curve["true_positive_rate"],
                "color": color,
            }
        )
        f1_series.append(
            {
                "label": _class_label(class_id, display_names),
                "x": np.asarray(curve["thresholds"])[::-1],
                "y": np.asarray(curve["f1"])[::-1],
                "color": color,
            }
        )

    plot_specs = (
        (config.PR_CURVE_PLOT_NAME, "Pixel-wise PR Curve", "Recall", "Precision", precision_series),
        (config.ROC_CURVE_PLOT_NAME, "Pixel-wise ROC Curve", "False Positive Rate", "True Positive Rate", roc_series),
        (config.F1_CURVE_PLOT_NAME, "Pixel-wise F1 Curve", "Confidence Threshold", "F1", f1_series),
    )
    for filename, title, x_label, y_label, series in plot_specs:
        canvas = np.full(
            (config.CURVE_PLOT_HEIGHT, config.CURVE_PLOT_WIDTH, 3),
            config.PLOT_BACKGROUND_BGR,
            dtype=np.uint8,
        )
        _draw_chart(
            canvas,
            (0, 0, config.CURVE_PLOT_WIDTH, config.CURVE_PLOT_HEIGHT),
            title,
            x_label,
            y_label,
            series,
        )
        cv2.imwrite(str(output_dir / filename), canvas)

    _save_confusion_matrix(
        np.asarray(confusion_matrix, dtype=np.int64),
        output_dir / config.CONFUSION_MATRIX_PLOT_NAME,
        display_names,
    )
    return curve_metrics


def _save_confusion_matrix(
    confusion: np.ndarray,
    output_path: Path,
    class_names: Tuple[str, ...],
) -> None:
    """绘制按真实类别归一化的像素级混淆矩阵。"""
    class_count = confusion.shape[0]
    canvas = np.full((900, 1000, 3), config.PLOT_BACKGROUND_BGR, dtype=np.uint8)
    cv2.putText(
        canvas, "Normalized Confusion Matrix", (250, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.PLOT_AXIS_BGR, 2, cv2.LINE_AA,
    )
    left, top, cell_size = 220, 150, min(145, 560 // max(class_count, 1))
    row_totals = np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    normalized = confusion / row_totals

    for class_id in range(class_count):
        label = _class_label(class_id, class_names)
        y_center = top + class_id * cell_size + cell_size // 2
        cv2.putText(
            canvas, label, (left - 125, y_center + 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, (left + class_id * cell_size + 14, top - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
        )
        for predicted_id in range(class_count):
            value = float(normalized[class_id, predicted_id])
            shade = int(round(255 * (1.0 - value)))
            color = (shade, shade, 255)
            x0 = left + predicted_id * cell_size
            y0 = top + class_id * cell_size
            cv2.rectangle(
                canvas, (x0, y0), (x0 + cell_size - 2, y0 + cell_size - 2),
                color, thickness=-1,
            )
            cv2.rectangle(
                canvas, (x0, y0), (x0 + cell_size - 2, y0 + cell_size - 2),
                config.PLOT_BACKGROUND_BGR, thickness=1,
            )
            cv2.putText(
                canvas, f"{value * 100:.1f}%", (x0 + 15, y0 + cell_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, f"n={int(confusion[class_id, predicted_id])}",
                (x0 + 15, y0 + cell_size // 2 + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (40, 40, 40), 1, cv2.LINE_AA,
            )

    cv2.putText(
        canvas, "Predicted class", (left + 170, top + class_count * cell_size + 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.PLOT_AXIS_BGR, 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)
