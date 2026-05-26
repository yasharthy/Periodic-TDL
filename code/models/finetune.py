# models/finetune.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

# uses your existing embedder (which wraps PolyHeteroModel and returns atom/graph reps)
from models.embedding import HSMPEmbedding


class HSMPFineTune(nn.Module):
    """
    Fine-tuning model:
      - uses only atom embeddings from HSMPEmbedding
      - mean-pools them to a graph representation (embedder already returns graph_repr)
      - MLP head for downstream prediction (regression/classification)
    """
    def __init__(self, embedder: HSMPEmbedding, out_dim: int = 1, hidden: Optional[int] = None, drop: float = 0.1):
        super().__init__()
        self.embedder = embedder
        H = hidden if hidden is not None else embedder.hidden

        # simple 2-layer head; tweak width/depth as needed
        self.head = nn.Sequential(
            nn.Linear(embedder.hidden, H),
            nn.LayerNorm(H),           # <-- added
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(H, out_dim),
        )

        self.out_dim = out_dim
        self.hidden = H

    def forward(self, data):
        emb = self.embedder(data)           # dict with: atom_repr, graph_repr, batch
        g = emb["graph_repr"]               # mean-pooled node embeddings
        yhat = self.head(g)                 # [B, out_dim]
        return yhat

    # GROVER-style param groups (smaller LR for encoder, larger LR for head)
    def param_groups(self, lr_encoder: float, lr_head: float, weight_decay: float = 0.0, freeze_encoder: bool = False):
        if freeze_encoder:
            for p in self.embedder.parameters():
                p.requires_grad = False
            return [
                {"params": self.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
            ]
        else:
            return [
                {"params": self.embedder.parameters(), "lr": lr_encoder, "weight_decay": weight_decay},
                {"params": self.head.parameters(),     "lr": lr_head,    "weight_decay": weight_decay},
            ]
