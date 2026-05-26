# model.py
# GROVER-style organization for HSMP pretraining:
# - HSMPEmbedding: wraps your encoder (PolyHeteroModel) and returns atom/bond/graph embeddings
# - Heads: AtomVocabHead, BondVocabHead, FGHead
# - GroverLikePretrainTask: builds sparse targets, computes CE/BCE losses, reports metrics

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

# Import your existing encoder
from models.embedding import HSMPEmbedding


# ----------
# Heads
# ----------

class AtomVocabHead(nn.Module):
    """Classifier over atom context vocabulary (class 0 reserved for ignore)."""
    def __init__(self, hidden: int, n_classes_with_ignore: int):
        super().__init__()
        self.fc = nn.Linear(hidden, n_classes_with_ignore)
    def forward(self, atom_repr: torch.Tensor) -> torch.Tensor:
        return self.fc(atom_repr)


class BondVocabHead(nn.Module):
    """Classifier over bond context vocabulary (class 0 reserved for ignore)."""
    def __init__(self, hidden: int, n_classes_with_ignore: int):
        super().__init__()
        self.fc = nn.Linear(hidden, n_classes_with_ignore)
    def forward(self, bond_repr: torch.Tensor) -> torch.Tensor:
        return self.fc(bond_repr)


class FGHead(nn.Module):
    """Graph-level multi-label predictor for functional-group bits."""
    def __init__(self, hidden: int, n_fg: int):
        super().__init__()
        self.fc = nn.Linear(hidden, n_fg)
    def forward(self, graph_repr: torch.Tensor) -> torch.Tensor:
        return self.fc(graph_repr)


# ------------------------------
# GROVER-like Pretraining Task
# ------------------------------

def _labels_from_tokens(tokens, vocab: Dict[str, int], p_keep: float, device: torch.device) -> torch.Tensor:
    """
    Sparse supervision à la GROVER:
      - With prob p_keep, keep real vocab id
      - Else 0 (ignore_index)
      - Unseen tokens -> <UNK> if present; else 0
    Returns: LongTensor [L]
    """
    L = len(tokens)
    # if L == 0:
    #     return torch.zeros(0, dtype=torch.long, device=device)
    unk = vocab["<UNK>"]
    # print(len(tokens))
    ids = torch.tensor([vocab.get(t, unk) for t in tokens], dtype=torch.long, device=device)

    keep = torch.rand(L, device=device) < p_keep
    out = torch.zeros_like(ids)
    out[keep] = ids[keep]
    return out


def _top1_on_labeled(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, int]:
    """Top-1 accuracy on non-ignored rows (target != 0)."""
    # if targets.numel() == 0:
    #     return 0.0, 0
    mask = targets != 0
    total = int(mask.sum().item())
    # if total == 0:
    #     return 0.0, 0
    preds = logits.argmax(dim=-1)
    correct = int((preds[mask] == targets[mask]).sum().item())
    return (correct / max(1, total)), total


class GroverLikePretrainTask(nn.Module):
    """
    GROVER-style pretraining on HSMP embeddings (single-stream):
      - Atom vocab CE on x_node[1]
      - Bond vocab CE on ea_f1 (directed edges of (node_k1,'via_edge',node_k1))
      - Optional FG BCE on pooled graph embeddings if data has `fg` (set w_fg>0)
    Total loss: w_atom * CE_atom + w_bond * CE_bond + w_fg * BCE_fg
    """
    def __init__(self,
                 embedder: HSMPEmbedding,
                 atom_vocab: Dict[str, int],
                 bond_vocab: Dict[str, int],
                 n_fg: int = 85,
                 w_atom: float = 1.0,
                 w_bond: float = 1.0,
                 w_fg:   float = 1.0,
                 p_atom: float = 0.15,
                 p_bond: float = 0.15):
        super().__init__()
        self.embedder = embedder
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.p_atom = p_atom
        self.p_bond = p_bond

        n_atom_cls = 1 + max(atom_vocab.values()) #if len(atom_vocab) else 0)
        n_bond_cls = 1 + max(bond_vocab.values()) #if len(bond_vocab) else 0)

        H = embedder.hidden
        self.atom_head = AtomVocabHead(H, n_atom_cls)
        self.bond_head = BondVocabHead(H, n_bond_cls)
        self.fg_head   = FGHead(H, n_fg) #if (n_fg is not None and n_fg > 0) else None

        self.ce_atom = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        self.ce_bond = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        self.bce_fg  = nn.BCEWithLogitsLoss(reduction="mean") #if self.fg_head is not None else None

        self.w_atom = w_atom
        self.w_bond = w_bond
        self.w_fg   = w_fg

    def _make_targets(self, data, device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Atom tokens at node_k1
        atom_tokens = getattr(data["node_k1"], "ctx_label", None)
        assert atom_tokens is not None, "ERROR! Expected atom vocab"
        atom_tokens = [item for sublist in atom_tokens for item in sublist]        
        y_atom = _labels_from_tokens(atom_tokens, self.atom_vocab, self.p_atom, device)

        # Bond tokens at directed edges (node_k1,'via_edge',node_k1)
        rel = ("node_k1", "via_edge", "node_k1")
        bond_tokens = getattr(data[rel], "ctx_label", None)
        assert bond_tokens is not None, "ERROR! Expected bond vocab"
        bond_tokens = [item for sublist in bond_tokens for item in sublist]        
        y_bond = _labels_from_tokens(bond_tokens, self.bond_vocab, self.p_bond, device)

        # FG multi-label targets
        y_fg = getattr(data, "fg", None)
        assert y_fg is not None, "ERROR! Expected FG attribute"
        y_fg = y_fg.to(device)
        return y_atom, y_bond, y_fg

    def forward(self, data, return_logits: bool = False) -> Dict:
        device = next(self.parameters()).device
        data = data.to(device)

        y_atom, y_bond, y_fg = self._make_targets(data, device)

        emb = self.embedder(data)
        atom_repr  = emb["atom_repr"]
        bond_repr  = emb["bond_repr"]
        graph_repr = emb["graph_repr"]

        # sanity checks (fast fail with clear message)
        # if y_atom.numel() and atom_repr.size(0) != y_atom.size(0):
        if atom_repr.size(0) != y_atom.size(0):
            raise RuntimeError(f"Atom target length ({y_atom.size(0)}) != #nodes ({atom_repr.size(0)}) in batch.")
        # if y_bond.numel() and bond_repr.size(0) != y_bond.size(0):
        if bond_repr.size(0) != y_bond.size(0):
            raise RuntimeError(f"Bond target length ({y_bond.size(0)}) != #directed-edges ({bond_repr.size(0)}) in batch.")

        atom_logits = self.atom_head(atom_repr)
        bond_logits = self.bond_head(bond_repr)
        fg_logits = self.fg_head(graph_repr)
        # Note! y_fg will be a 1D tensor because of the way data is collated in PyG. So we reshape
        y_fg = y_fg.view(fg_logits.size(0), fg_logits.size(1))

        # print(atom_logits.shape, bond_logits.shape, fg_logits.shape)
        # print(y_atom.shape, y_bond.shape, y_fg.shape)

        # Losses
        loss_atom = self.ce_atom(atom_logits, y_atom) #if y_atom.numel() else torch.tensor(0.0, device=device)
        loss_bond = self.ce_bond(bond_logits, y_bond) #if y_bond.numel() else torch.tensor(0.0, device=device)
        loss_fg = self.bce_fg(fg_logits, y_fg.float())

        loss = self.w_atom * loss_atom + self.w_bond * loss_bond + self.w_fg * loss_fg

        # Metrics on labeled positions only
        a_acc, a_cnt = _top1_on_labeled(atom_logits, y_atom)
        b_acc, b_cnt = _top1_on_labeled(bond_logits, y_bond)

        out = {
            "loss": loss,
            "loss_parts": {
                "atom_ce": float(loss_atom.item()),
                "bond_ce": float(loss_bond.item()),
                "fg_bce":  float(loss_fg.item()),
            },
            "metrics": {
                "atom_top1": a_acc, "n_atom_labeled": a_cnt,
                "bond_top1": b_acc, "n_bond_labeled": b_cnt,
            },
        }
        if return_logits:
            out["logits"] = {"atom": atom_logits, "bond": bond_logits, "fg": fg_logits}
        return out

    # GROVER-style param groups (smaller LR for encoder, larger for heads)
    def param_groups(self, lr_encoder: float, lr_heads: float, weight_decay: float = 0.0):
        enc_params  = list(self.embedder.parameters())
        head_params = list(self.atom_head.parameters()) + list(self.bond_head.parameters()) + list(self.fg_head.parameters())
        return [
            {"params": enc_params,  "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_heads,   "weight_decay": weight_decay},
        ]
