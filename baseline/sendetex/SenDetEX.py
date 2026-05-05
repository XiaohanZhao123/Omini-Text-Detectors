import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureEncoder(nn.Module):
    def __init__(self, proxy_model_embed, vocab_size):
        super().__init__()
        self.proxy_embed = proxy_model_embed
        self.vocab_size = vocab_size
    def forward(self, sentence, regenerated, token_probs, token_logits):
        # Paper Section 3.1: "z_ins = M_proxy^embed(s_i) in R^{1 x d}" —
        # sentence-level embedding, so mean-pool over the token dimension.
        e_s = self.proxy_embed(sentence)         # (L_s, d)
        e_r = self.proxy_embed(regenerated)      # (L_r, d)
        z_ins = e_s.mean(dim=0, keepdim=True)    # (1, d)
        z_inf = e_r.mean(dim=0, keepdim=True)    # (1, d)
        probs = F.softmax(token_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (L_s,)
        return token_probs, entropy, z_ins, z_inf
class StyleExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=d_model, kernel_size=5, groups=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        )
        # Paper describes a linear projection after the Transformer to map
        # features into R^{Li x d}. The input S is (Li, 2); the Transformer
        # expects d_model features, so we project 2 -> d_model before it.
        # This is equivalent to "pre-projection + transformer" which matches
        # the paper's functional intent (Section 3.2, Global Branch).
        self.global_proj_in = nn.Linear(2, d_model)
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True), num_layers=2
        )
        self.gate = nn.Linear(2 * d_model, d_model)
    def forward(self, token_probs, entropies):
        # token_probs: (Li,), entropies: (Li,)
        S = torch.stack([token_probs, entropies], dim=-1)  # (Li, 2)
        # Local branch: Conv1D expects (B, C, L)
        S_trans = S.transpose(0, 1).unsqueeze(0)           # (1, 2, Li)
        local_feat = self.local_conv(S_trans).squeeze(0).transpose(0, 1)  # (Li, d)
        # Global branch: project 2 -> d, run transformer, strip batch dim
        S_g = self.global_proj_in(S).unsqueeze(0)          # (1, Li, d)
        global_feat = self.global_transformer(S_g).squeeze(0)  # (Li, d)
        concat = torch.cat([local_feat, global_feat], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        fused = gate * local_feat + (1 - gate) * global_feat
        z_style = fused.mean(dim=0, keepdim=True)          # (1, d)
        return z_style
class TripleCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn3 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.classifier = nn.Linear(d_model, 1)
    def forward(self, z_style, z_ins, z_inf):
        # Every input is (1, d) per the paper (Section 3.3). Add a batch
        # axis to get (1, 1, d) for MultiheadAttention(batch_first=True).
        def _to_3d(x):
            if x.dim() == 2:
                return x.unsqueeze(0)  # (1, d) -> (1, 1, d)
            return x
        zs, zi, zf = _to_3d(z_style), _to_3d(z_ins), _to_3d(z_inf)
        a1, _ = self.attn1(zs, zi, zf)    # (1, 1, d)
        a2, _ = self.attn2(zi, zf, zs)    # (1, 1, d)
        a3, _ = self.attn3(zf, zs, zi)    # (1, 1, d)
        cross = torch.cat([a1, a2, a3], dim=-1)       # (1, 1, 3d)
        z_cross = self.fusion(cross)                  # (1, 1, d)
        p = torch.sigmoid(self.classifier(z_cross))   # (1, 1, 1)
        return p.squeeze()                             # scalar
class SenDetEX(nn.Module):
    def __init__(self, proxy_model_embed, vocab_size, d_model=768):
        super().__init__()
        self.encoder = FeatureEncoder(proxy_model_embed, vocab_size)
        self.style = StyleExtractor(d_model)
        self.fusion = TripleCrossAttention(d_model)
        self.loss_fn = nn.MSELoss()
    def forward(self, s_i, r_i, token_probs, token_logits, label):
        p_i, e_i, z_ins, z_inf = self.encoder(s_i, r_i, token_probs, token_logits)
        z_style = self.style(p_i, e_i)
        pred = self.fusion(z_style, z_ins, z_inf)
        loss = self.loss_fn(pred.squeeze(), label)
        return pred, loss

