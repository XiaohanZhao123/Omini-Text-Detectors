"""Shared utilities for evaluation scripts."""

from evaluate.data_loader import EvalRecord


def format_output_record(
    eval_record: EvalRecord, detector_name: str, detection_result: dict
) -> dict:
    """Format a detection result into the standard output schema.

    Args:
        eval_record: Ground-truth evaluation record.
        detector_name: Name of the detector that produced the result.
        detection_result: Raw result dict from the detector pipeline.

    Returns:
        Standardized output dict suitable for JSONL serialization.
    """
    predicted_label = detection_result["label"]
    ground_truth = eval_record.ground_truth_label

    return {
        "detection": {
            "detector": detector_name,
            "label": predicted_label,
            "correct": predicted_label == ground_truth,
            "detector_metadata": detection_result.get("metadata", {}),
        },
        "ground_truth": {"label": ground_truth},
        "reference": {
            "source_file": eval_record.source_file,
            "line_index": eval_record.line_index,
            "text_field": eval_record.text_field,
        },
        "metadata": {
            "domain": eval_record.domain,
            "task": eval_record.task,
            "ai_model": eval_record.ai_model,
        },
        "score": detection_result.get("score", None),
    }
