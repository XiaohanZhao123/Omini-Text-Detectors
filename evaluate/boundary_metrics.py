"""Boundary detection evaluation metrics.

Provides metrics for evaluating AI text boundary detection at multiple levels:
- Token-level: Per-word classification metrics (accuracy, precision, recall, F1)
- Span-level: IoU for detected AI spans
- Boundary-level: MAE for boundary position
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def token_level_metrics(
    pred_labels: List[int],
    true_labels: List[int],
) -> Dict[str, float]:
    """Calculate token-level classification metrics.

    Args:
        pred_labels: Predicted per-word labels (0=human, 1=AI)
        true_labels: Ground truth per-word labels

    Returns:
        Dict with accuracy, precision, recall, f1, and per-class metrics
    """
    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Length mismatch: pred={len(pred_labels)}, true={len(true_labels)}"
        )

    if not pred_labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "human_precision": 0.0,
            "human_recall": 0.0,
            "ai_precision": 0.0,
            "ai_recall": 0.0,
        }

    pred = np.array(pred_labels)
    true = np.array(true_labels)

    # Overall accuracy
    accuracy = np.mean(pred == true)

    # AI class metrics (positive class)
    tp_ai = np.sum((pred == 1) & (true == 1))
    fp_ai = np.sum((pred == 1) & (true == 0))
    fn_ai = np.sum((pred == 0) & (true == 1))

    ai_precision = tp_ai / (tp_ai + fp_ai) if (tp_ai + fp_ai) > 0 else 0.0
    ai_recall = tp_ai / (tp_ai + fn_ai) if (tp_ai + fn_ai) > 0 else 0.0
    ai_f1 = (
        2 * ai_precision * ai_recall / (ai_precision + ai_recall)
        if (ai_precision + ai_recall) > 0
        else 0.0
    )

    # Human class metrics
    tp_human = np.sum((pred == 0) & (true == 0))
    fp_human = np.sum((pred == 0) & (true == 1))
    fn_human = np.sum((pred == 1) & (true == 0))

    human_precision = tp_human / (tp_human + fp_human) if (tp_human + fp_human) > 0 else 0.0
    human_recall = tp_human / (tp_human + fn_human) if (tp_human + fn_human) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(ai_precision),  # AI precision (positive class)
        "recall": float(ai_recall),  # AI recall (positive class)
        "f1": float(ai_f1),  # AI F1 (positive class)
        "human_precision": float(human_precision),
        "human_recall": float(human_recall),
        "ai_precision": float(ai_precision),
        "ai_recall": float(ai_recall),
    }


def span_iou(
    pred_intervals: List[List[int]],
    true_intervals: List[List[int]],
) -> float:
    """Calculate Intersection over Union for span intervals.

    Args:
        pred_intervals: Predicted AI intervals [[start, end], ...]
        true_intervals: Ground truth AI intervals [[start, end], ...]

    Returns:
        IoU score (0 to 1)
    """
    if not pred_intervals and not true_intervals:
        return 1.0  # Both empty = perfect match
    if not pred_intervals or not true_intervals:
        return 0.0  # One empty = no overlap

    def intervals_to_set(intervals: List[List[int]]) -> set:
        """Convert intervals to set of positions."""
        positions = set()
        for start, end in intervals:
            positions.update(range(start, end))
        return positions

    pred_set = intervals_to_set(pred_intervals)
    true_set = intervals_to_set(true_intervals)

    intersection = len(pred_set & true_set)
    union = len(pred_set | true_set)

    return intersection / union if union > 0 else 0.0


def boundary_mae(
    pred_boundary: Optional[int],
    true_boundary: Optional[int],
    text_length: int,
) -> float:
    """Calculate Mean Absolute Error for boundary position.

    Args:
        pred_boundary: Predicted boundary word index (first AI word)
        true_boundary: Ground truth boundary word index
        text_length: Total number of words in text

    Returns:
        MAE normalized by text length (0 to 1)
    """
    if pred_boundary is None and true_boundary is None:
        return 0.0  # Both no boundary = correct
    if pred_boundary is None:
        pred_boundary = text_length  # No prediction = boundary at end
    if true_boundary is None:
        true_boundary = text_length  # No true boundary = all human

    error = abs(pred_boundary - true_boundary)
    return error / text_length if text_length > 0 else 0.0


def get_first_boundary(
    word_labels: List[int],
) -> Optional[int]:
    """Get the first word index where AI content begins.

    Args:
        word_labels: Per-word labels (0=human, 1=AI)

    Returns:
        Index of first AI word, or None if all human
    """
    for i, label in enumerate(word_labels):
        if label == 1:
            return i
    return None


def evaluate_boundary_detection(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Dict[str, float]:
    """Comprehensive boundary detection evaluation.

    Args:
        predictions: List of dicts with 'word_labels', 'ai_char_intervals', 'ai_word_intervals'
        ground_truths: List of dicts with same structure

    Returns:
        Dict with aggregated metrics across all samples
    """
    token_metrics_list = []
    iou_scores = []
    mae_scores = []

    for pred, truth in zip(predictions, ground_truths):
        # Token-level
        pred_labels = pred.get("word_labels", [])
        true_labels = truth.get("word_labels", [])

        if pred_labels and true_labels:
            # Pad shorter list if needed
            max_len = max(len(pred_labels), len(true_labels))
            pred_labels = pred_labels + [0] * (max_len - len(pred_labels))
            true_labels = true_labels + [0] * (max_len - len(true_labels))

            metrics = token_level_metrics(pred_labels, true_labels)
            token_metrics_list.append(metrics)

        # Span IoU (character-level)
        pred_char_int = pred.get("ai_char_intervals", [])
        true_char_int = truth.get("ai_char_intervals", [])
        iou = span_iou(pred_char_int, true_char_int)
        iou_scores.append(iou)

        # Boundary MAE (word-level)
        pred_boundary = get_first_boundary(pred.get("word_labels", []))
        true_boundary = get_first_boundary(truth.get("word_labels", []))
        text_len = len(truth.get("word_labels", []))
        mae = boundary_mae(pred_boundary, true_boundary, text_len)
        mae_scores.append(mae)

    # Aggregate metrics
    result = {
        "num_samples": len(predictions),
    }

    # Token-level aggregates
    if token_metrics_list:
        for key in token_metrics_list[0].keys():
            values = [m[key] for m in token_metrics_list]
            result[f"token_{key}"] = float(np.mean(values))
            result[f"token_{key}_std"] = float(np.std(values))

    # Span IoU aggregate
    if iou_scores:
        result["span_iou"] = float(np.mean(iou_scores))
        result["span_iou_std"] = float(np.std(iou_scores))

    # Boundary MAE aggregate
    if mae_scores:
        result["boundary_mae"] = float(np.mean(mae_scores))
        result["boundary_mae_std"] = float(np.std(mae_scores))

    return result


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """Format metrics as a human-readable report.

    Args:
        metrics: Dict from evaluate_boundary_detection

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "BOUNDARY DETECTION EVALUATION REPORT",
        "=" * 60,
        f"Samples evaluated: {metrics.get('num_samples', 0)}",
        "",
        "TOKEN-LEVEL METRICS (per-word classification):",
        f"  Accuracy:   {metrics.get('token_accuracy', 0):.4f} +/- {metrics.get('token_accuracy_std', 0):.4f}",
        f"  AI Precision: {metrics.get('token_ai_precision', 0):.4f} +/- {metrics.get('token_ai_precision_std', 0):.4f}",
        f"  AI Recall:    {metrics.get('token_ai_recall', 0):.4f} +/- {metrics.get('token_ai_recall_std', 0):.4f}",
        f"  AI F1:        {metrics.get('token_f1', 0):.4f} +/- {metrics.get('token_f1_std', 0):.4f}",
        "",
        "SPAN-LEVEL METRICS (AI interval detection):",
        f"  IoU:       {metrics.get('span_iou', 0):.4f} +/- {metrics.get('span_iou_std', 0):.4f}",
        "",
        "BOUNDARY-LEVEL METRICS (transition point):",
        f"  MAE:       {metrics.get('boundary_mae', 0):.4f} +/- {metrics.get('boundary_mae_std', 0):.4f}",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the metrics
    print("Testing boundary metrics...")

    # Test token-level metrics
    pred_labels = [0, 0, 0, 1, 1, 1, 1, 0, 0]
    true_labels = [0, 0, 1, 1, 1, 1, 0, 0, 0]
    metrics = token_level_metrics(pred_labels, true_labels)
    print(f"Token metrics: {metrics}")

    # Test span IoU
    pred_intervals = [[3, 7]]  # positions 3-6
    true_intervals = [[2, 6]]  # positions 2-5
    iou = span_iou(pred_intervals, true_intervals)
    print(f"Span IoU: {iou}")  # Should be 3/5 = 0.6

    # Test boundary MAE
    mae = boundary_mae(3, 2, 9)
    print(f"Boundary MAE: {mae}")  # Should be 1/9 = 0.111

    # Test comprehensive evaluation
    predictions = [
        {"word_labels": pred_labels, "ai_char_intervals": pred_intervals},
    ]
    ground_truths = [
        {"word_labels": true_labels, "ai_char_intervals": true_intervals},
    ]
    result = evaluate_boundary_detection(predictions, ground_truths)
    print("\nComprehensive evaluation:")
    print(format_metrics_report(result))
