import sys
import json
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from sklearn.metrics import classification_report, cohen_kappa_score
from utils.config import GLCLiCModelConfig
from transformers import AutoTokenizer
from transformers import AutoModel
from torch.optim import AdamW



class GLCLiC(L.LightningModule):

    def __init__(self, config: GLCLiCModelConfig, tokenizer_len: int):
        super(GLCLiC, self).__init__()

        self.lr = config.lr
        self.alpha = config.alpha

        self.margin = 0.1
        self.sub_margin = lambda z: z - self.margin

        self.shared_backbone_model = AutoModel.from_pretrained('microsoft/deberta-v3-base', attention_probs_dropout_prob=config.dropout, hidden_dropout_prob=config.dropout)
        self.shared_backbone_model.resize_token_embeddings(tokenizer_len)

        trainable_layers = [
            "embeddings.word_embeddings",
            "embeddings.LayerNorm",
            "encoder.layer.11",
            "encoder.LayerNorm"
        ]

        for name, params in self.shared_backbone_model.named_parameters():
            if any(layer in name for layer in trainable_layers):
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.shared_backbone_model.training = True

        feature_dims = [768]

        if config.global_coherence:
            self.global_coherence_linear = nn.Linear(768, 1)
            feature_dims.append(768)

        if config.local_coherence:
            self.local_coherence_linear = nn.Linear(768, 1)
            feature_dims.append(768)

        if config.global_lexical:
            self.global_lexical_mlp = nn.Linear(512, 768)
            self.global_lexical_linear = nn.Linear(768, 1)
            feature_dims.append(768)

        if config.local_lexical:
            self.local_lexical_linear = nn.Linear(768, 1)
            feature_dims.append(768)


        additional_special_tokens = ["[GLOBAL COHERENCE]", "[LOCAL COHERENCE]", "[GLOBAL LEXICAL]", "[LOCAL LEXICAL]", "[REPRESENTATION]"]
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

        self.final_feature_norm = nn.LayerNorm(768*len(feature_dims))
        self.regression_criterion = nn.MSELoss()


        if config.dataset == "CoAuthor":
            self.classifier = nn.Linear(768*len(feature_dims), 3)
            self.criterion = nn.CrossEntropyLoss()

            self.validation_step_output = {
                "targets": [],
                "predictions": [],
                "type": []
            }

            self.test_step_output = {
                "text": [],
                "text_len": [],
                "targets": [],
                "predictions": [],
                "type": []
            }


        if config.dataset == "SeqXGPT-Bench":
            self.classifier = nn.Linear(768*len(feature_dims), 1)
            self.criterion = nn.BCEWithLogitsLoss()\

            self.validation_step_output = {
                "targets": [],
                "predictions": [],
                "probability": [],
            }

            self.test_step_output = {
                "text": [],
                "targets": [],
                "predictions": [],
                "probability": [],
                "source": [],
            }


        self.config = config


    def calculate_coauthor_metrics(self, targets, predictions):
        cls_report = classification_report(targets, predictions, output_dict=True, zero_division=0)
        kappa_score = cohen_kappa_score(targets, predictions)

        # 0: Human, 1: AI, 2: Human-AI

        accuracy = cls_report["accuracy"]
        macro_f1_score = cls_report["macro avg"]["f1-score"]
        weighted_f1_score = cls_report["weighted avg"]["f1-score"]

        try:
            human_precision = cls_report["0"]["precision"]
            human_recall = cls_report["0"]["recall"]
            human_f1_score = cls_report["0"]["f1-score"]
        except:
            human_precision = -1
            human_recall = -1
            human_f1_score = -1

        try:
            ai_precision = cls_report["1"]["precision"]
            ai_recall = cls_report["1"]["recall"]
            ai_f1_score = cls_report["1"]["f1-score"]
        except:
            ai_precision = -1
            ai_recall = -1
            ai_f1_score = -1

        try:
            human_ai_precision = cls_report["2"]["precision"]
            human_ai_recall = cls_report["2"]["recall"]
            human_ai_f1_score = cls_report["2"]["f1-score"]
        except:
            human_ai_precision = -1
            human_ai_recall = -1
            human_ai_f1_score = -1

        return accuracy, macro_f1_score, weighted_f1_score, human_precision, human_recall, human_f1_score, ai_precision, ai_recall, ai_f1_score, human_ai_precision, human_ai_recall, human_ai_f1_score, kappa_score


    def calculate_binary_metrics(self, targets, predictions, probs):
        targets = np.array(targets).astype(np.int64)
        predictions = np.array(predictions)

        cls_report = classification_report(targets, predictions, output_dict=True, zero_division=0)
        kappa_score = cohen_kappa_score(targets, predictions)

        try:
            auroc_score = roc_auc_score(targets, probs)
        except:
            print("AUROC Score Error: Returning 0")
            auroc_score = 0

        # 0: Human, 1: AI

        accuracy = cls_report["accuracy"]
        macro_f1_score = cls_report["macro avg"]["f1-score"]
        weighted_f1_score = cls_report["weighted avg"]["f1-score"]

        try:
            human_precision = cls_report["0"]["precision"]
            human_recall = cls_report["0"]["recall"]
            human_f1_score = cls_report["0"]["f1-score"]
        except:
            human_precision = -1
            human_recall = -1
            human_f1_score = -1

        try:
            ai_precision = cls_report["1"]["precision"]
            ai_recall = cls_report["1"]["recall"]
            ai_f1_score = cls_report["1"]["f1-score"]
        except:
            ai_precision = -1
            ai_recall = -1
            ai_f1_score = -1

        return accuracy, macro_f1_score, weighted_f1_score, human_precision, human_recall, human_f1_score, ai_precision, ai_recall, ai_f1_score, kappa_score, auroc_score


    def configure_optimizers(self):
        trainable_params = [
            {"params": self.shared_backbone_model.parameters(), "lr": 2e-5}, # Keep the differential LR if desired
            {"params": self.classifier.parameters(), "lr": self.lr},
        ]

        if self.config.global_coherence:
            trainable_params.append({"params": self.global_coherence_linear.parameters(), "lr": self.lr})

        if self.config.local_coherence:
            trainable_params.append({"params": self.local_coherence_linear.parameters(), "lr": self.lr})

        if self.config.global_lexical:
            trainable_params.append({"params": self.global_lexical_mlp.parameters(), "lr": self.lr})
            trainable_params.append({"params": self.global_lexical_linear.parameters(), "lr": self.lr})

        if self.config.local_lexical:
            trainable_params.append({"params": self.local_lexical_linear.parameters(), "lr": self.lr})

        optimizer = AdamW(trainable_params, weight_decay=0.01)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }


    def decode_input(self, sentence_rep_input):
        return self.tokenizer.batch_decode(sentence_rep_input[:, 0], skip_special_tokens=True)


    def contrastive_loss(self, pos_score, neg_scores):
        neg_scores_sub = torch.stack(list(map(self.sub_margin, neg_scores)))
        all_scores = torch.cat((neg_scores_sub, pos_score), dim=-1)
        lsmax = -1 * F.log_softmax(all_scores, dim=-1)
        pos_loss = lsmax[-1]
        return pos_loss


    def coherence_loss(self, additional_outputs):
        final_global_coherence_loss = 0
        final_local_coherence_loss = 0

        if "global_coherent_output" in additional_outputs:
            global_coherents_output = additional_outputs["global_coherent_output"]
            global_incocherents_outputs = additional_outputs["global_incocherent_outputs"]

            global_coherence_loss = []

            for global_coherent_output, global_incocherent_output in zip(global_coherents_output, global_incocherents_outputs):
                global_coherence_loss.append(self.contrastive_loss(global_coherent_output.unsqueeze(0), global_incocherent_output))

            final_global_coherence_loss = torch.stack(global_coherence_loss).mean()


        if "local_coherent_output" in additional_outputs:
            local_coherent_output = additional_outputs["local_coherent_output"]
            local_incocherent_outputs = additional_outputs["local_incocherent_outputs"]

            local_coherence_loss = []

            for local_coherent_output, local_incocherent_output in zip(local_coherent_output, local_incocherent_outputs):
                local_coherence_loss.append(self.contrastive_loss(local_coherent_output.unsqueeze(0), local_incocherent_output))

            final_local_coherence_loss = torch.stack(local_coherence_loss).mean()

        return final_global_coherence_loss, final_local_coherence_loss


    def lexical_loss(self, additional_outputs):
        final_global_lexical_loss = 0
        final_local_lexical_loss = 0

        if "global_lexical_output" in additional_outputs:
            global_lexical_output = additional_outputs["global_lexical_output"]
            global_lexical_target = additional_outputs["global_lexical_target"].float()

            global_lexical_loss = self.regression_criterion(global_lexical_output, global_lexical_target)
            final_global_lexical_loss = global_lexical_loss


        if "local_lexical_output" in additional_outputs:
            local_lexical_output = additional_outputs["local_lexical_output"]
            local_lexical_target = additional_outputs["local_lexical_target"].float()

            local_lexical_loss = self.regression_criterion(local_lexical_output, local_lexical_target)
            final_local_lexical_loss = local_lexical_loss

        return final_global_lexical_loss, final_local_lexical_loss


    def text_representation(self, input):
        representations = self.shared_backbone_model(input_ids=input[:, 0], attention_mask=input[:, 1], token_type_ids=input[:, 2], return_dict=True)
        return representations.last_hidden_state[:, 0, :]


    def forward(self, inputs, stage="train"):

        features = []
        addtional_outputs = {}

        sentence_representations = self.text_representation(inputs['sentence_inputs'])
        features.append(sentence_representations)

        if self.config.global_coherence:
            coherent_doc_input, incoherent_docs_input = inputs["global_coherences"]
            global_coherence_representation = self.text_representation(coherent_doc_input)
            global_coherent_output = self.global_coherence_linear(global_coherence_representation).view(-1)

            if stage == "test":
                global_incocherent_outputs = None
            else:
                global_incocherent_outputs = []
                for incoherent_doc_input in incoherent_docs_input:
                    global_incocherent_representation = self.text_representation(incoherent_doc_input)
                    global_incocherent_output = self.global_coherence_linear(global_incocherent_representation).view(-1)
                    global_incocherent_outputs.append(global_incocherent_output.squeeze())
                global_incocherent_outputs = torch.stack(global_incocherent_outputs)

            features.append(global_coherence_representation)
            addtional_outputs["global_coherent_output"] = global_coherent_output
            addtional_outputs["global_incocherent_outputs"] = global_incocherent_outputs

        if self.config.local_coherence:
            coherent_triplet_input, incoherent_triplets_input = inputs["local_coherences"]
            local_coherence_representation = self.text_representation(coherent_triplet_input)
            local_coherent_output = self.local_coherence_linear(local_coherence_representation).view(-1)

            if stage == "test":
                local_incocherent_outputs = None
            else:
                local_incocherent_outputs = []
                for incoherent_triplet_input in incoherent_triplets_input:
                    local_incoherent_representation = self.text_representation(incoherent_triplet_input)
                    local_incocherent_output = self.local_coherence_linear(local_incoherent_representation).view(-1)
                    local_incocherent_outputs.append(local_incocherent_output.squeeze())
                local_incocherent_outputs = torch.stack(local_incocherent_outputs)

            features.append(local_coherence_representation)
            addtional_outputs["local_coherent_output"] = local_coherent_output
            addtional_outputs["local_incocherent_outputs"] = local_incocherent_outputs

        if self.config.global_lexical:
            global_lexical_features = inputs["global_lexical"]
            global_lexical_features = self.global_lexical_mlp(global_lexical_features)
            global_lexical_output = self.global_lexical_linear(global_lexical_features).view(-1)

            features.append(global_lexical_features)
            addtional_outputs["global_lexical_output"] = global_lexical_output
            addtional_outputs["global_lexical_target"] = inputs["global_cefr_level"]

        if self.config.local_lexical:
            batched_local_cefr_levels, cefr_input = inputs["local_lexical"]
            cefr_sentence_idxs = cefr_input[:, 3]
            cefr_representation = self.shared_backbone_model(input_ids=cefr_input[:, 0], attention_mask=cefr_input[:, 1], token_type_ids=cefr_input[:, 2])
            cefr_representations = cefr_representation.last_hidden_state

            local_cefr_representations = []

            for local_cefr_levels, cefr_sentence_idx, cefr_representation in zip(batched_local_cefr_levels, cefr_sentence_idxs, cefr_representations):
                local_cefr_levels = local_cefr_levels[local_cefr_levels != -1]

                mask_cefr_level = torch.isin(cefr_sentence_idx, local_cefr_levels)
                select_sentence_idx = torch.nonzero(mask_cefr_level, as_tuple=True)[0]

                cefr_representation = torch.index_select(cefr_representation, 0, select_sentence_idx)
                local_cefr_representations.append(cefr_representation.mean(axis=0))

            local_cefr_representations = torch.stack(local_cefr_representations)
            local_cefr_output = self.local_lexical_linear(local_cefr_representations).view(-1)

            features.append(local_cefr_representations)
            addtional_outputs["local_lexical_output"] = local_cefr_output
            addtional_outputs["local_lexical_target"] = inputs["local_cefr_level"]

        features = torch.cat(features, dim=-1)
        features = self.final_feature_norm(features)
        output = self.classifier(features)

        return output, addtional_outputs


    def training_step(self, batch, batch_idx):
        output, addtional_outputs = self(batch)

        if self.config.dataset == "SeqXGPT-Bench":
            if output.shape[-1] == 1:
                output = output.squeeze(dim=-1)

        global_coherence_loss, local_coherence_loss = self.coherence_loss(addtional_outputs)
        global_lexical_loss, local_lexical_loss = self.lexical_loss(addtional_outputs)
        target_loss = self.criterion(output, batch["labels"].float())

        loss = target_loss + self.alpha * (global_coherence_loss + local_coherence_loss + global_lexical_loss + local_lexical_loss)

        self.log_dict({
            'train_loss': loss,
            'train_global_coherence_loss': global_coherence_loss,
            'train_local_coherence_loss': local_coherence_loss,
            'train_global_lexical_loss': global_lexical_loss,
            'train_local_lexical_loss': local_lexical_loss,
            'train_target_loss': target_loss
            }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):

        output, addtional_outputs = self(batch)

        if self.config.dataset == "SeqXGPT-Bench":
            if output.shape[-1] == 1:
                output = output.squeeze(dim=-1)

        global_coherence_loss, local_coherence_loss = self.coherence_loss(addtional_outputs)
        global_lexical_loss, local_lexical_loss = self.lexical_loss(addtional_outputs)
        target_loss = self.criterion(output, batch["labels"].float())

        loss = target_loss + self.alpha * (global_coherence_loss + local_coherence_loss + global_lexical_loss + local_lexical_loss)

        if self.config.dataset == "CoAuthor":
            targets = torch.argmax(batch["labels"], dim=1).cpu().numpy()
            predictions = torch.argmax(output, dim=1).cpu().numpy()

            self.validation_step_output["targets"].extend(targets)
            self.validation_step_output["predictions"].extend(predictions)
            self.validation_step_output["type"].extend(batch["types"].cpu().numpy())

        if self.config.dataset == "SeqXGPT-Bench":
            output_probs = torch.nn.functional.sigmoid(output)

            targets = batch['labels'].squeeze().cpu().numpy()
            predictions = torch.where(output_probs > 0.5, 1, 0).squeeze().cpu().numpy()
            output_probs = output_probs.squeeze().cpu().numpy()

            if targets.shape == ():
                targets = np.array([targets.item()])

            if predictions.shape == ():
                predictions = np.array([predictions.item()])

            if output_probs.shape == ():
                output_probs = np.array([output_probs.item()])

            self.validation_step_output["targets"].extend(targets)
            self.validation_step_output["predictions"].extend(predictions)
            self.validation_step_output["probability"].extend(output_probs)

        self.log_dict({
            'val_loss': loss,
            'val_global_coherence_loss': global_coherence_loss,
            'val_local_coherence_loss': local_coherence_loss,
            'val_global_lexical_loss': global_lexical_loss,
            'val_local_lexical_loss': local_lexical_loss,
            'val_target_loss': target_loss
            }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def on_validation_epoch_end(self):
        if self.config.dataset == "CoAuthor":
            argumentative_idx = [idx for idx, essay_type in enumerate(self.validation_step_output["type"]) if essay_type == 1]
            creative_idx = [idx for idx, essay_type in enumerate(self.validation_step_output["type"]) if essay_type == 0]

            argumentative_targets = [self.validation_step_output["targets"][idx] for idx in argumentative_idx]
            argumentative_predictions = [self.validation_step_output["predictions"][idx] for idx in argumentative_idx]

            creative_targets = [self.validation_step_output["targets"][idx] for idx in creative_idx]
            creative_predictions = [self.validation_step_output["predictions"][idx] for idx in creative_idx]

            all_targets = self.validation_step_output["targets"]
            all_predictions = self.validation_step_output["predictions"]

            argumentative_metrics = self.calculate_coauthor_metrics(argumentative_targets, argumentative_predictions)
            creative_metrics = self.calculate_coauthor_metrics(creative_targets, creative_predictions)
            combined_metrics = self.calculate_coauthor_metrics(all_targets, all_predictions)

            argumentative_accuracy, argumentative_macro_f1_score, argumentative_weighted_f1_score, argumentative_human_precision, argumentative_human_recall, argumentative_human_f1_score, argumentative_ai_precision, argumentative_ai_recall, argumentative_ai_f1_score, argumentative_human_ai_precision, argumentative_human_ai_recall, argumentative_human_ai_f1_score, argumentative_kappa_score = argumentative_metrics
            creative_accuracy, creative_macro_f1_score, creative_weighted_f1_score, creative_human_precision, creative_human_recall, creative_human_f1_score, creative_ai_precision, creative_ai_recall, creative_ai_f1_score, creative_human_ai_precision, creative_human_ai_recall, creative_human_ai_f1_score, creative_kappa_score = creative_metrics
            accuracy, macro_f1_score, weighted_f1_score, human_precision, human_recall, human_f1_score, ai_precision, ai_recall, ai_f1_score, human_ai_precision, human_ai_recall, human_ai_f1_score, kappa_score = combined_metrics


            self.log_dict({
                "val_kappa_score": kappa_score,
                "val_accuracy": accuracy,
                "val_macro_f1_score": macro_f1_score,
                "val_weighted_f1_score": weighted_f1_score,
                "val_human_precision": human_precision,
                "val_human_recall": human_recall,
                "val_human_f1_score": human_f1_score,
                "val_ai_precision": ai_precision,
                "val_ai_recall": ai_recall,
                "val_ai_f1_score": ai_f1_score,
                "val_human_ai_precision": human_ai_precision,
                "val_human_ai_recall": human_ai_recall,
                "val_human_ai_f1_score": human_ai_f1_score,
                "val_argumentative_accuracy": argumentative_accuracy,
                "val_argumentative_macro_f1_score": argumentative_macro_f1_score,
                "val_argumentative_weighted_f1_score": argumentative_weighted_f1_score,
                "val_argumentative_human_precision": argumentative_human_precision,
                "val_argumentative_human_recall": argumentative_human_recall,
                "val_argumentative_human_f1_score": argumentative_human_f1_score,
                "val_argumentative_ai_precision": argumentative_ai_precision,
                "val_argumentative_ai_recall": argumentative_ai_recall,
                "val_argumentative_ai_f1_score": argumentative_ai_f1_score,
                "val_argumentative_human_ai_precision": argumentative_human_ai_precision,
                "val_argumentative_human_ai_recall": argumentative_human_ai_recall,
                "val_argumentative_human_ai_f1_score": argumentative_human_ai_f1_score,
                "val_argumentative_kappa_score": argumentative_kappa_score,
                "val_creative_accuracy": creative_accuracy,
                "val_creative_macro_f1_score": creative_macro_f1_score,
                "val_creative_weighted_f1_score": creative_weighted_f1_score,
                "val_creative_human_precision": creative_human_precision,
                "val_creative_human_recall": creative_human_recall,
                "val_creative_human_f1_score": creative_human_f1_score,
                "val_creative_ai_precision": creative_ai_precision,
                "val_creative_ai_recall": creative_ai_recall,
                "val_creative_ai_f1_score": creative_ai_f1_score,
                "val_creative_human_ai_precision": creative_human_ai_precision,
                "val_creative_human_ai_recall": creative_human_ai_recall,
                "val_creative_human_ai_f1_score": creative_human_ai_f1_score,
                "val_creative_kappa_score": creative_kappa_score,
                }, prog_bar=True, logger=True)

        if self.config.dataset == "SeqXGPT-Bench":
            all_targets = self.validation_step_output["targets"]
            all_predictions = self.validation_step_output["predictions"]
            all_probs = self.validation_step_output["probability"]

            all_metrics = self.calculate_binary_metrics(all_targets, all_predictions, all_probs)
            accuracy, macro_f1_score, weighted_f1_score, human_precision, human_recall, human_f1_score, ai_precision, ai_recall, ai_f1_score, kappa_score, auroc_score = all_metrics

            self.log_dict({
                "val_kappa_score": kappa_score,
                "val_auroc_score": auroc_score,
                "val_accuracy": accuracy,
                "val_macro_f1_score": macro_f1_score,
                "val_weighted_f1_score": weighted_f1_score,
                "val_human_precision": human_precision,
                "val_human_recall": human_recall,
                "val_human_f1_score": human_f1_score,
                "val_ai_precision": ai_precision,
                "val_ai_recall": ai_recall,
                "val_ai_f1_score": ai_f1_score,
                }, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        output, _ = self(batch, stage="test")

        if self.config.dataset == "CoAuthor":
            targets = torch.argmax(batch["labels"], dim=1).cpu().numpy()
            predictions = torch.argmax(output, dim=1).cpu().numpy()

            texts = self.decode_input(batch["sentence_inputs"])
            texts_len = [len(text) for text in texts]

            self.test_step_output["text"].extend(texts)
            self.test_step_output["text_len"].extend(texts_len)
            self.test_step_output["targets"].extend(targets)
            self.test_step_output["predictions"].extend(predictions)
            self.test_step_output["type"].extend(batch["types"].cpu().numpy())

        if self.config.dataset == "SeqXGPT-Bench":
            if output.shape[-1] == 1:
                output = output.squeeze(dim=-1)

            output_probs = torch.nn.functional.sigmoid(output)

            targets = batch['labels'].squeeze().cpu().numpy()
            predictions = torch.where(output_probs > 0.5, 1, 0).squeeze().cpu().numpy()
            output_probs = output_probs.squeeze().cpu().numpy()

            if targets.shape == ():
                targets = np.array([targets.item()])

            if predictions.shape == ():
                predictions = np.array([predictions.item()])

            if output_probs.shape == ():
                output_probs = np.array([output_probs.item()])

            texts = self.decode_input(batch["sentence_inputs"])

            self.test_step_output["text"].extend(texts)
            self.test_step_output["targets"].extend(targets)
            self.test_step_output["predictions"].extend(predictions)
            self.test_step_output["probability"].extend(output_probs)
            self.test_step_output["source"].extend(batch["sources"].cpu().numpy())


    def on_test_epoch_end(self):
        if self.config.dataset == "CoAuthor":
            argumentative_idx = [idx for idx, essay_type in enumerate(self.test_step_output["type"]) if essay_type == 1]
            creative_idx = [idx for idx, essay_type in enumerate(self.test_step_output["type"]) if essay_type == 0]

            argumentative_targets = [self.test_step_output["targets"][idx] for idx in argumentative_idx]
            argumentative_predictions = [self.test_step_output["predictions"][idx] for idx in argumentative_idx]

            creative_targets = [self.test_step_output["targets"][idx] for idx in creative_idx]
            creative_predictions = [self.test_step_output["predictions"][idx] for idx in creative_idx]

            all_targets = self.test_step_output["targets"]
            all_predictions = self.test_step_output["predictions"]

            argumentative_metrics = self.calculate_coauthor_metrics(argumentative_targets, argumentative_predictions)
            creative_metrics = self.calculate_coauthor_metrics(creative_targets, creative_predictions)
            combined_metrics = self.calculate_coauthor_metrics(all_targets, all_predictions)

            argumentative_accuracy, argumentative_macro_f1_score, argumentative_weighted_f1_score, argumentative_human_precision, argumentative_human_recall, argumentative_human_f1_score, argumentative_ai_precision, argumentative_ai_recall, argumentative_ai_f1_score, argumentative_human_ai_precision, argumentative_human_ai_recall, argumentative_human_ai_f1_score, argumentative_kappa_score = argumentative_metrics
            creative_accuracy, creative_macro_f1_score, creative_weighted_f1_score, creative_human_precision, creative_human_recall, creative_human_f1_score, creative_ai_precision, creative_ai_recall, creative_ai_f1_score, creative_human_ai_precision, creative_human_ai_recall, creative_human_ai_f1_score, creative_kappa_score = creative_metrics
            accuracy, macro_f1_score, weighted_f1_score, human_precision, human_recall, human_f1_score, ai_precision, ai_recall, ai_f1_score, human_ai_precision, human_ai_recall, human_ai_f1_score, kappa_score = combined_metrics

            test_df = {
                "text": self.test_step_output["text"],
                "text_len": self.test_step_output["text_len"],
                "targets": np.array(self.test_step_output["targets"]).tolist(),
                "predictions": np.array(self.test_step_output["predictions"]).tolist(),
                "type": torch.tensor(self.test_step_output["type"]).tolist(),
            }

            test_df = pd.DataFrame(test_df)
            test_df_csv_path = f"logs/test_results_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
            test_df.to_csv(test_df_csv_path, index=False)

            self.log_dict({
                "global_coherence_feature": 1 if self.config.global_coherence else 0,
                "local_coherence_feature": 1 if self.config.local_coherence else 0,
                "global_lexical_feature": 1 if self.config.global_lexical else 0,
                "local_lexical_feature": 1 if self.config.local_lexical else 0,
                "test_accuracy": accuracy,
                "test_macro_f1_score": macro_f1_score,
                "test_weighted_f1_score": weighted_f1_score,
                "test_human_precision": human_precision,
                "test_human_recall": human_recall,
                "test_human_f1_score": human_f1_score,
                "test_ai_precision": ai_precision,
                "test_ai_recall": ai_recall,
                "test_ai_f1_score": ai_f1_score,
                "test_human_ai_precision": human_ai_precision,
                "test_human_ai_recall": human_ai_recall,
                "test_human_ai_f1_score": human_ai_f1_score,
                "test_kappa_score": kappa_score,
                "test_argumentative_accuracy": argumentative_accuracy,
                "test_argumentative_macro_f1_score": argumentative_macro_f1_score,
                "test_argumentative_weighted_f1_score": argumentative_weighted_f1_score,
                "test_argumentative_human_precision": argumentative_human_precision,
                "test_argumentative_human_recall": argumentative_human_recall,
                "test_argumentative_human_f1_score": argumentative_human_f1_score,
                "test_argumentative_ai_precision": argumentative_ai_precision,
                "test_argumentative_ai_recall": argumentative_ai_recall,
                "test_argumentative_ai_f1_score": argumentative_ai_f1_score,
                "test_argumentative_human_ai_precision": argumentative_human_ai_precision,
                "test_argumentative_human_ai_recall": argumentative_human_ai_recall,
                "test_argumentative_human_ai_f1_score": argumentative_human_ai_f1_score,
                "test_argumentative_kappa_score": argumentative_kappa_score,
                "test_creative_accuracy": creative_accuracy,
                "test_creative_macro_f1_score": creative_macro_f1_score,
                "test_creative_weighted_f1_score": creative_weighted_f1_score,
                "test_creative_human_precision": creative_human_precision,
                "test_creative_human_recall": creative_human_recall,
                "test_creative_human_f1_score": creative_human_f1_score,
                "test_creative_ai_precision": creative_ai_precision,
                "test_creative_ai_recall": creative_ai_recall,
                "test_creative_ai_f1_score": creative_ai_f1_score,
                "test_creative_human_ai_precision": creative_human_ai_precision,
                "test_creative_human_ai_recall": creative_human_ai_recall,
                "test_creative_human_ai_f1_score": creative_human_ai_f1_score,
                "test_creative_kappa_score": creative_kappa_score,
                }, prog_bar=True, logger=True)

        if self.config.dataset == "SeqXGPT-Bench":
            all_targets = self.test_step_output["targets"]
            all_predictions = self.test_step_output["predictions"]
            all_probs = self.test_step_output["probability"]

            all_metrics = self.calculate_binary_metrics(all_targets, all_predictions, all_probs)
            accuracy, macro_f1_score, weighted_f1_score, human_precision, human_recall, human_f1_score, ai_precision, ai_recall, ai_f1_score, kappa_score, auroc_score = all_metrics

            test_df = {
                "text": self.test_step_output["text"],
                "targets": np.array(self.test_step_output["targets"]).tolist(),
                "predictions": np.array(self.test_step_output["predictions"]).tolist(),
                "probability": np.array(self.test_step_output["probability"]).tolist(),
                "source": np.array(self.test_step_output["source"]).tolist(),
            }

            test_df = pd.DataFrame(test_df)
            test_df_csv_path = f"logs/test_results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            test_df.to_csv(test_df_csv_path, index=False)

            self.log_dict({
                "global_coherence_feature": 1 if self.config.global_coherence else 0,
                "local_coherence_feature": 1 if self.config.local_coherence else 0,
                "global_lexical_feature": 1 if self.config.global_lexical else 0,
                "local_lexical_feature": 1 if self.config.local_lexical else 0,
                "test_accuracy": accuracy,
                "test_macro_f1_score": macro_f1_score,
                "test_weighted_f1_score": weighted_f1_score,
                "test_human_precision": human_precision,
                "test_human_recall": human_recall,
                "test_human_f1_score": human_f1_score,
                "test_ai_precision": ai_precision,
                "test_ai_recall": ai_recall,
                "test_ai_f1_score": ai_f1_score,
                "test_kappa_score": kappa_score,
                "test_auroc_score": auroc_score,
                }, prog_bar=True, logger=True)
