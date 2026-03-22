import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class RoBERTaModernBERTCRF(nn.Module):
    """
    Model that fuses RoBERTa and ModernBERT outputs, style features, and a CRF layer for sequence labeling.
    """
    def __init__(self, roberta_model_name, modernbert_model_name, num_labels, style_feature_dim=4, style_hidden_size=64):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = AutoModel.from_pretrained(roberta_model_name, use_safetensors=True)
        self.modernbert = AutoModel.from_pretrained(modernbert_model_name, use_safetensors=True)
        roberta_hidden_size = self.roberta.config.hidden_size
        modernbert_hidden_size = self.modernbert.config.hidden_size
        self.fusion_layer = nn.Linear(roberta_hidden_size + modernbert_hidden_size, roberta_hidden_size)
        self.style_projector = nn.Linear(style_feature_dim, style_hidden_size)
        self.style_attention = nn.MultiheadAttention(style_hidden_size, 4, batch_first=True)
        self.classifier = nn.Linear(roberta_hidden_size + style_hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def compute_info_mask(self, style_features, attention_mask):
        style_hidden = torch.relu(self.style_projector(style_features))
        attn_output, _ = self.style_attention(style_hidden, style_hidden, style_hidden, key_padding_mask=~attention_mask.bool())
        info_mask = torch.sigmoid(attn_output.sum(dim=-1)) * attention_mask
        return info_mask

    def forward(self, input_ids, attention_mask, style_features):
        roberta_out = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        modernbert_out = self.modernbert(input_ids, attention_mask=attention_mask).last_hidden_state
        combined = torch.cat([roberta_out, modernbert_out], dim=-1)
        fused = torch.relu(self.fusion_layer(combined))
        fused = self.dropout(fused)
        info_mask = self.compute_info_mask(style_features, attention_mask)
        fused = fused * info_mask.unsqueeze(-1)
        style_hidden = torch.relu(self.style_projector(style_features))
        combined_final = torch.cat([fused, style_hidden], dim=-1)
        logits = self.classifier(combined_final)
        mask = attention_mask.bool()
        # Handle different CRF library versions
        # pytorch-crf uses 'decode', torchcrf uses 'viterbi_decode'
        if hasattr(self.crf, 'decode'):
            preds = self.crf.decode(logits, mask=mask)
        else:
            preds = self.crf.viterbi_decode(logits, mask=mask)
        return preds, info_mask, logits