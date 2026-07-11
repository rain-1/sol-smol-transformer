from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


TOKENS = list("0123456789=")
PAD_TOKEN = "<pad>"
VOCAB = TOKENS + [PAD_TOKEN]
PAD_ID = VOCAB.index(PAD_TOKEN)
OPERATION_TOKENS = {
    "copy": "<copy>", "reverse": "<reverse>", "sort": "<sort>",
    "rotate_left": "<rotate_left>",
}
LETTER_TOKENS = list("abc")
# Keep existing operation IDs stable (12–15) so earlier multi-task checkpoints load.
OPERATION_TOKENS["translate"] = "<translate>"
MULTITASK_VOCAB = VOCAB + list(OPERATION_TOKENS.values())[:4] + LETTER_TOKENS + [OPERATION_TOKENS["translate"]]
OPERATION_IDS = {name: MULTITASK_VOCAB.index(token) for name, token in OPERATION_TOKENS.items()}
LETTER_IDS = {letter: MULTITASK_VOCAB.index(letter) for letter in LETTER_TOKENS}
INPUT_TOKENS = TOKENS + LETTER_TOKENS


@dataclass
class ModelConfig:
    seq_len: int = 8
    vocab_size: int = len(VOCAB)
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.0

    def dict(self) -> dict:
        return asdict(self)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, capture: bool = False, padding_mask: torch.Tensor | None = None):
        normed = self.ln1(x)
        attn_out, weights = self.attn(
            normed, normed, normed, key_padding_mask=padding_mask,
            need_weights=capture, average_attn_weights=False
        )
        x = x + self.dropout(attn_out)
        mlp_pre = self.mlp[0](self.ln2(x))
        mlp_activation = self.mlp[1](mlp_pre)
        x = x + self.dropout(self.mlp[2](mlp_activation))
        if capture:
            return x, {"attention": weights, "mlp": mlp_activation}
        return x, None


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, tokens: torch.Tensor, capture: bool = False):
        if tokens.shape[1] > self.cfg.seq_len:
            raise ValueError(f"Sequence exceeds maximum length {self.cfg.seq_len}")
        padding_mask = tokens.eq(PAD_ID)
        x = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        internals = []
        for block in self.blocks:
            x, layer_data = block(x, capture=capture, padding_mask=padding_mask)
            if capture:
                internals.append(layer_data)
        return self.output(self.final_norm(x)), internals


def encode(text: str, seq_len: int, pad: bool = False) -> torch.Tensor:
    if not text.endswith("="):
        text += "="
    if len(text) > seq_len or any(c not in INPUT_TOKENS for c in text) or text.count("=") != 1:
        raise ValueError(f"Input must use digits or a/b/c followed by =, at most {seq_len} tokens")
    if pad:
        ids = [MULTITASK_VOCAB.index(c) for c in text] + [PAD_ID] * (seq_len - len(text))
        return torch.tensor(ids, dtype=torch.long)
    return torch.tensor([MULTITASK_VOCAB.index(c) for c in text], dtype=torch.long)


def decode(tokens: torch.Tensor, vocab: list[str] | None = None) -> str:
    if vocab is None:
        vocab = MULTITASK_VOCAB if any(int(i) >= len(VOCAB) for i in tokens) else VOCAB
    return "".join(vocab[int(i)] for i in tokens if int(i) != PAD_ID)
