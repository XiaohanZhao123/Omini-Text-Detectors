"""
Extracted model classes from GenAI_Detect_Sentence_Level notebooks.

Paper: Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation
       (arXiv:2509.17830, EMNLP 2025)
Source: github.com/saitejalekkala33/GenAI_Detect_Sentence_Level

Architecture: Transformer encoder → BiGRU → LayerNorm → MLP → Linear → CRF
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class DeBERTaBiGRUCRFTagger(nn.Module):
    """
    DeBERTa + BiGRU + CRF for token-level AI text detection.

    Architecture:
        DeBERTa encoder → Dropout → BiGRU → LayerNorm → MLP → Linear → CRF

    Args:
        model_name: HuggingFace model ID (e.g. 'microsoft/deberta-v3-base')
        num_labels: Number of output labels (default: 2, human/AI)
        hidden_dim: Hidden dimension for BiGRU (default: 512)
        num_layers: Number of GRU layers (default: 2)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self, model_name, num_labels=2, hidden_dim=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.deberta.config.hidden_size
        self.gru = nn.GRU(
            hidden_size, hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.hidden2hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # Xavier init for classification head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            labels: (B, L) token labels (0=human, 1=AI, -100=ignore)
                    If provided, returns CRF loss. Otherwise, returns predictions.

        Returns:
            If labels provided: scalar loss
            If no labels: (B, L) tensor of predicted labels
        """
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        gru_out, _ = self.gru(sequence_output)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.hidden2hidden(gru_out)
        logits = self.classifier(gru_out)

        if labels is not None:
            mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0  # CRF needs valid labels
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            return loss, logits
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            # Pad predictions to full sequence length
            padded = []
            for pred in predictions:
                pad_len = attention_mask.size(1) - len(pred)
                padded.append(pred + [0] * pad_len)
            return torch.tensor(padded, device=input_ids.device), logits
