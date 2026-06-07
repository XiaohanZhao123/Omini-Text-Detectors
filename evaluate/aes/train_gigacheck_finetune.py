"""
Fine-tune GigaCheck DETR on AES Chains.

This is a modified copy of gigacheck/train/scripts/train_detr_model.py.
Changes from official:
  1. load_model(): skip reset_parameters() to preserve pretrained DETR weights
  2. compute_metrics(): add token-level AI F1 for checkpoint selection

Usage:
    cd <REPO_ROOT>
    bash draft/train_gigacheck_aes.sh
"""

import pathlib
import re
from typing import Dict, List, Tuple
from loguru import logger
from functools import partial
import os

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoConfig
from transformers.trainer_utils import set_seed, EvalPrediction

from gigacheck.train.src.detection.arguments_parsing import DataArguments, ModelArguments, TrainingArguments
from gigacheck.train.src.detection.interval_dataset import IntervalEncodedDataset
from gigacheck.train.src.detection.detr_trainer import DetrTrainer

from gigacheck.train.src.data.corpus import Corpus
from gigacheck.train.src.data.data_format import TextSample
from gigacheck.train.src.data.utils import Input
from gigacheck.train.src.utils import print_trainable_parameters, save_config

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification
from gigacheck.model.src.interval_detector.metrics.intervals_metrics import AveragePrecision
from gigacheck.model.src.interval_detector.metrics.sentence_metrics import SentenceMetrics
from gigacheck.model.src.interval_detector.span_utils import gt_to_absolete, convert_outputs


# =============================================================================
# Token-level metrics helpers
# =============================================================================

CONF_INTERVAL_THRESH = 0.8  # Same threshold as GigaCheck inference

def intervals_to_word_labels(text, ai_char_intervals):
    """Convert character-level AI intervals to per-word binary labels.

    Uses >50% overlap threshold, same as GigaCheck's intervals_to_word_labels().
    Returns list of 0 (human) / 1 (ai) per word.
    """
    words = list(re.finditer(r'\S+', text))
    labels = []
    for w in words:
        w_start, w_end = w.start(), w.end()
        w_len = w_end - w_start
        max_overlap = 0
        for ai_start, ai_end in ai_char_intervals:
            overlap = max(0, min(w_end, ai_end) - max(w_start, ai_start))
            max_overlap = max(max_overlap, overlap)
        labels.append(1 if max_overlap / w_len > 0.5 else 0)
    return labels


def compute_token_level_metrics(pred_labels, true_labels):
    """Compute token-level AI precision, recall, F1."""
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = np.mean(pred == true)
    return {"accuracy": accuracy, "ai_precision": precision, "ai_recall": recall, "ai_f1": f1}


# =============================================================================
# compute_metrics — CHANGED: adds token-level AI F1 for checkpoint selection
# =============================================================================

def compute_metrics(eval_pred: EvalPrediction, eval_dataset: IntervalEncodedDataset) -> Dict[str, float]:
    all_pred_spans, all_pred_logits = eval_pred.predictions
    all_text_lengths, all_text_inds = eval_pred.label_ids
    assert len(all_text_lengths) == len(all_pred_logits) == len(all_pred_spans) == len(all_text_inds)

    map_metric = AveragePrecision()
    sentence_metric = SentenceMetrics()

    # Accumulators for token-level metrics
    all_token_pred = []
    all_token_true = []

    for pred_logits, pred_spans, text_length, text_ind in zip(
        all_pred_logits, all_pred_spans, all_text_lengths, all_text_inds,
    ):
        pred_spans = torch.tensor(pred_spans)
        pred_logits = torch.tensor(pred_logits)
        span_labels = eval_dataset.__getitem__(text_ind).span_labels

        gt_spans = gt_to_absolete(span_labels, text_length) if len(span_labels) > 0 else span_labels
        preds = convert_outputs(pred_logits, pred_spans, text_length)

        # --- mAP (unchanged) ---
        map_metric.update(
            [{"pred_relevant_windows": preds, "text_id": text_ind}],
            [{"relevant_windows": gt_spans, "text_id": text_ind}],
            text_length
        )
        sample = eval_dataset.texts[text_ind]
        sentence_metric.update(preds, gt_spans, sample.text, sample.sep_indices)

        # --- Token-level metrics (NEW) ---
        # Filter predicted intervals by confidence threshold (same as inference)
        confident_preds = preds[preds[:, 2] > CONF_INTERVAL_THRESH].numpy() if len(preds) > 0 else np.empty((0, 4))
        pred_char_intervals = [[float(row[0]), float(row[1])] for row in confident_preds]

        # Ground truth: convert sample's ai_char_intervals to word labels
        gt_char_intervals = sample.ai_char_intervals if sample.ai_char_intervals else []

        pred_word_labels = intervals_to_word_labels(sample.text, pred_char_intervals)
        true_word_labels = intervals_to_word_labels(sample.text, gt_char_intervals)

        all_token_pred.extend(pred_word_labels)
        all_token_true.extend(true_word_labels)

    # mAP metrics
    map_metrics = map_metric.compute()
    out_dict = {}
    for name, value in map_metrics.items():
        if "mAP" in name:
            out_dict[f"eval_{name}"] = float(value)
    out_dict.update(sentence_metric.to_dict("eval"))

    # Token-level metrics (NEW)
    token_metrics = compute_token_level_metrics(all_token_pred, all_token_true)
    out_dict["eval_token_accuracy"] = token_metrics["accuracy"]
    out_dict["eval_token_ai_precision"] = token_metrics["ai_precision"]
    out_dict["eval_token_ai_recall"] = token_metrics["ai_recall"]
    out_dict["eval_token_ai_f1"] = token_metrics["ai_f1"]

    return out_dict


# =============================================================================
# load_datasets — UNCHANGED from official
# =============================================================================

def load_datasets(
    data_args: DataArguments,
    tokenizer: transformers.PreTrainedTokenizer,
    id2label: Dict[int, str],
    random_seed: int = 0,
    span_loss_type: str = "l1",
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    train_texts: List[TextSample] = Corpus(data_args.train_data_path).data
    val_texts: List[TextSample] = Corpus(data_args.eval_data_path).data

    train_dataset = IntervalEncodedDataset(
        train_texts,
        tokenizer,
        data_args.max_sequence_length,
        data_args.min_sequence_length,
        random=data_args.random_sequence_length,
        id2label=id2label,
        seed=random_seed,
        span_loss_type=span_loss_type,
    )

    eval_dataset = IntervalEncodedDataset(
        val_texts,
        tokenizer,
        id2label=id2label,
        span_loss_type=span_loss_type,
        is_eval=True,
    )

    return train_dataset, eval_dataset


# =============================================================================
# update_tokenizer — UNCHANGED from official
# =============================================================================

def update_tokenizer(tokenizer, max_sequence_length):
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.max_len = max_sequence_length
    tokenizer.padding_side = "right"


# =============================================================================
# load_model — CHANGED: skip reset_parameters() for fine-tuning
# =============================================================================

def load_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> MistralAIDetectorForSequenceClassification:

    if training_args.bf16:
        raise NotImplementedError("BF16 is not yet implemented. You can set --extractor_dtype 'bfloat16'.")

    kwargs = {"token": model_args.hf_token} if model_args.hf_token else {}
    arch_name = AutoConfig.from_pretrained(model_args.pretrained_model_name, **kwargs).architectures[-1]

    # Load pretrained model (backbone + DETR weights)
    model = MistralAIDetectorForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_args.pretrained_model_name,
        with_detr=True,
        freeze_backbone=True,
        # Don't pass detr_config — use the saved config from the pretrained checkpoint
        # so the DETR architecture matches the saved weights exactly.
        # detr_config=model_args.to_dict(),  # <-- REMOVED for fine-tuning
        torch_dtype=torch.float32,
        num_labels=len(model_args.id2label),
        id2label=model_args.id2label,
        max_length=tokenizer.max_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    if getattr(torch, model_args.extractor_dtype) == torch.bfloat16:
        logger.info(f"Use bfloat16 type for model.model ({type(model.model)})")
        model.model.to(torch.bfloat16)
        model.classification_head.to(torch.bfloat16)

    if arch_name != "MistralAIDetectorForSequenceClassification":
        model.classification_head = None
        model.config.trained_classification_head = False
        print("Remove classification_head from model.")

    # =========================================================================
    # CHANGE: Skip reset_parameters() to preserve pretrained DETR weights.
    #
    # Official code does:
    #     for module in model.detr.modules():
    #         if hasattr(module, "reset_parameters"):
    #             module.reset_parameters()
    #         if hasattr(module, "_reset_parameters"):
    #             module._reset_parameters()
    #     model.detr.reset_parameters()
    #
    #     if model.config.detr_config.get("special_ref_points", False):
    #         model.detr.transformer.init_special_ref_points()
    #
    # We skip ALL of this for fine-tuning.
    # =========================================================================
    logger.info("Fine-tuning mode: keeping pretrained DETR weights (no reset_parameters).")

    print_trainable_parameters(model)
    return model


# =============================================================================
# train — UNCHANGED from official (except calls our load_model)
# =============================================================================

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    kwargs = {"token": model_args.hf_token} if model_args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name, **kwargs)
    update_tokenizer(tokenizer, data_args.max_sequence_length)

    train_dataset, eval_dataset = load_datasets(
        data_args,
        tokenizer,
        model_args.id2label,
        random_seed=training_args.seed,
        span_loss_type=model_args.span_loss_type,
    )

    model = load_model(model_args, training_args, tokenizer)

    save_config(model.config, model_args.pretrained_model_name, model_args.id2label, training_args.output_dir)
    logger.info(f"***** Running DETR fine-tuning for {training_args.num_train_epochs} epochs *****")

    log_extra_losses = list(model.criterion.weight_dict.keys())
    trainer = DetrTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Input.collate_fn,
        custom_id2label=model_args.id2label,
        compute_metrics=partial(compute_metrics, eval_dataset=eval_dataset),
        log_extra_losses=log_extra_losses,
        tokenizer_to_save=tokenizer,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
