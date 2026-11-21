"""Small decoder-only Transformer (GPT-2 style) with explicit parameter-count
conventions and an explicit per-token FLOPs formula.

The parameter-count conventions are the object of study in this project, so they
are defined in ONE place (`count_params`) and used everywhere else.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling.sizes import VOCAB_SIZE, flops_per_token as _flops


@dataclass
class GPTConfig:
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    seq_len: int = 512
    vocab_size: int = VOCAB_SIZE


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.out = nn.Linear(4 * c.d_model, c.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)  # input token embedding
        self.wpe = nn.Embedding(c.seq_len, c.d_model)  # learned positions
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # UNTIED output embedding
        self.apply(self._init)
        for b in self.blocks:  # scaled init of residual projections
            nn.init.normal_(b.proj.weight, std=0.02 / (2 * c.n_layer) ** 0.5)
            nn.init.normal_(b.out.weight, std=0.02 / (2 * c.n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))


def count_params(model: GPT) -> dict:
    """All parameter-count conventions, as raw group sizes plus named conventions.

    NOTE: CONVENTIONS below are our working definitions. Align them with the exact
    definitions used by Schaeffer et al. (2025) before running the final analysis.
    """
    n = lambda mods: sum(p.numel() for m in mods for p in m.parameters())
    emb_in = n([model.wte, model.wpe])
    head = n([model.lm_head])
    body = n(list(model.blocks) + [model.ln_f])
    return {
        "emb_in": emb_in,
        "head": head,
        "body": body,
        # --- named conventions ---
        "total": emb_in + head + body,  # everything
        "nonembed": body,  # transformer blocks only
        "nonembed_plus_head": body + head,  # everything except the INPUT embeddings
    }


CONVENTIONS = ["total", "nonembed", "nonembed_plus_head"]


def flops_per_token(c: GPTConfig) -> float:
    """Training FLOPs per token (fwd+bwd = 3x fwd).

    6 * (matmul params: 12*L*d^2 in blocks + d*V in the LM head)
    + 6 * L * T * d   (causal attention score/value matmuls; Kaplan-style)
    Embedding lookups cost no FLOPs.  Formula lives in sizes.py (torch-free).
    """
    return _flops(c.n_layer, c.d_model, c.seq_len, c.vocab_size)
