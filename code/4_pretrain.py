#!/usr/bin/env python
# 4_pretrain_stream.py
import os, json, gc, random, argparse
from pathlib import Path
from glob import glob
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader as PYGDataLoader

# Project modules
from models.embedding import HSMPEmbedding
from models.pretrain import GroverLikePretrainTask
# from utils.helpers import sanitize_feature_dims  # we'll wrap this with a schema variant

# ---------------- helpers: train / eval (unchanged logic) ----------------
def train_epoch(task, loader, device, grad_clip=5.0):
    task.train()
    opt = task._optimizer
    total_loss = 0.0
    n_batches = 0
    a_correct = 0.0
    a_total   = 0
    b_correct = 0.0
    b_total   = 0

    for batch in loader:
        batch = batch.to(device)

        opt.zero_grad(set_to_none=True)
        out = task(batch)
        loss = out["loss"]
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(task.parameters(), grad_clip)
        opt.step()

        n_atom_labeled = out["metrics"]["n_atom_labeled"]
        n_bond_labeled = out["metrics"]["n_bond_labeled"]
        assert n_atom_labeled > 0, "Batch had zero labeled atoms (increase p_atom or inspect data)."
        assert n_bond_labeled > 0, "Batch had zero labeled bonds (increase p_bond or inspect data)."

        total_loss += float(loss.item())
        n_batches  += 1

        a_correct += out["metrics"]["atom_top1"] * n_atom_labeled
        a_total   += n_atom_labeled
        b_correct += out["metrics"]["bond_top1"] * n_bond_labeled
        b_total   += n_bond_labeled

    assert n_batches > 0, "No batches in train loader."
    assert a_total > 0 and b_total > 0, "No labeled examples seen in training epoch."

    atom_acc = a_correct / a_total
    bond_acc = b_correct / b_total
    return total_loss / n_batches, atom_acc, a_total, bond_acc, b_total


@torch.no_grad()
def eval_epoch(task, loader, device):
    task.eval()
    total_loss = 0.0
    n_batches = 0
    a_correct = 0.0
    a_total   = 0
    b_correct = 0.0
    b_total   = 0

    for batch in loader:
        batch = batch.to(device)
        out = task(batch)
        total_loss += float(out["loss"].item())
        n_batches  += 1

        n_atom_labeled = out["metrics"]["n_atom_labeled"]
        n_bond_labeled = out["metrics"]["n_bond_labeled"]
        assert n_atom_labeled > 0, "Val batch had zero labeled atoms."
        assert n_bond_labeled > 0, "Val batch had zero labeled bonds."

        a_correct += out["metrics"]["atom_top1"] * n_atom_labeled
        a_total   += n_atom_labeled
        b_correct += out["metrics"]["bond_top1"] * n_bond_labeled
        b_total   += n_bond_labeled

    assert n_batches > 0, "No batches in val loader."
    assert a_total > 0 and b_total > 0, "No labeled examples seen in validation epoch."

    atom_acc = a_correct / a_total
    bond_acc = b_correct / b_total
    return total_loss / n_batches, atom_acc, a_total, bond_acc, b_total


# ---------------- constants ----------------
HETERO_DIR = "data/HeteroGraphsPretrain/pretrain_1M_v1"
PT_CKPT_DIR = "ckpt_pretrain"
VOCAB_DIR = "data/CtxVocab"
SCHEMA_PATH = Path(VOCAB_DIR) / "feature_schema.json"  # persisted global rel_F


# ---------------- schema utilities ----------------
def scan_global_rel_F(part_paths, max_parts=None):
    """
    One-time schema scan across shards: compute global max edge_attr width per relation.
    Loads each shard briefly, records widths, and frees memory immediately.
    """
    rel_F = defaultdict(int)
    count = 0
    for p in part_paths:
        if max_parts is not None and count >= max_parts:
            break
        graphs = torch.load(p, map_location="cpu")
        for g in graphs:
            for rel in g.edge_types:
                if 'edge_attr' in g[rel]:
                    F = int(g[rel].edge_attr.size(-1))
                    if F > rel_F[rel]:
                        rel_F[rel] = F
        del graphs
        gc.collect()
        count += 1
    # ensure minimum width = 1 (like your sanitize helper)
    for rel in list(rel_F.keys()):
        rel_F[rel] = max(1, int(rel_F[rel]))
    return {str(rel): int(w) for rel, w in rel_F.items()}  # make JSON-serializable keys


def sanitize_feature_dims_with_schema(dataset, rel_F):
    """
    Schema-aware version: pad/truncate each graph to the *global* feature widths (rel_F).
    Mirrors your sanitize_feature_dims behavior except it *does not* recompute maxima from the shard.
    """
    for g in dataset:
        for rel in g.edge_types:
            if 'edge_attr' in g[rel]:
                ea = g[rel].edge_attr
                key = str(rel)
                Fexp = int(rel_F.get(key, max(1, ea.size(-1))))  # fallback to self width if missing
                if ea.size(-1) == 0:
                    g[rel].edge_attr = torch.zeros((ea.size(0), Fexp), dtype=torch.float32, device=ea.device)
                elif ea.size(-1) < Fexp:
                    pad = torch.zeros((ea.size(0), Fexp - ea.size(1)), dtype=ea.dtype, device=ea.device)
                    g[rel].edge_attr = torch.cat([ea, pad], dim=1)
                elif ea.size(-1) > Fexp:
                    g[rel].edge_attr = ea[:, :Fexp]
    return dataset


def list_all_parts(root, stem="pretrain_1M"):
    """
    Finds all part files like: <root>/<stem>_hetero_partXX.pt
    Returns sorted list for determinism.
    """
    pattern = str(Path(root) / f"{stem}_hetero_part*.pt")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard files matched: {pattern}")
    return files


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Checkpoint directory name inside PT_CKPT_DIR, e.g. pretrain_1M_stream")
    ap.add_argument("--n_parts", required=True, type=int, help="Total number of chunks to use (e.g., 100)")

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--p_atom", type=float, default=0.15)
    ap.add_argument("--p_bond", type=float, default=0.15)

    ap.add_argument("--lr_encoder", type=float, default=2e-4)
    ap.add_argument("--lr_heads",   type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0001)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Streaming-specific knobs
    ap.add_argument("--val_parts", type=int, default=5, help="Number of shards held out for validation (fixed)")
    ap.add_argument("--val_ids", type=str, default="", help="Comma-separated shard indices for validation (optional, overrides val_parts)")
    ap.add_argument("--stem", type=str, default="pretrain_1M", help="Shard filename stem")
    args = ap.parse_args()

    # strict config checks
    assert 0.0 < args.p_atom <= 1.0, "p_atom must be in (0,1]."
    assert 0.0 < args.p_bond <= 1.0, "p_bond must be in (0,1]."

    random.seed(args.seed); torch.manual_seed(args.seed)

    # ---------------- discover shards ----------------
    all_parts = list_all_parts(HETERO_DIR, stem=args.stem)
    if args.n_parts is not None:
        all_parts = all_parts[:args.n_parts]
    n_total = len(all_parts)
    assert n_total >= args.val_parts, "n_parts must be >= val_parts"

    # ---------------- build / load global schema ----------------
    if SCHEMA_PATH.exists():
        rel_F = json.loads(Path(SCHEMA_PATH).read_text())
    else:
        rel_F = scan_global_rel_F(all_parts)
        SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        Path(SCHEMA_PATH).write_text(json.dumps(rel_F, indent=2))

    # ---------------- choose validation shards (fixed, deterministic) ----------------
    if args.val_ids.strip():
        val_ids = [int(t) for t in args.val_ids.split(",")]
        assert all(0 <= i < n_total for i in val_ids), "val_ids out of range"
        val_parts = [all_parts[i] for i in val_ids]
    else:
        # simple spaced pick for determinism (works because chunk creation was already randomized)
        step = max(1, n_total // args.val_parts)
        val_ids = list(range(0, step * args.val_parts, step))[:args.val_parts]
        val_parts = [all_parts[i] for i in val_ids]

    train_parts = [p for i, p in enumerate(all_parts) if i not in set(val_ids)]

    # ---------------- build fixed validation loader ----------------
    val_graphs = []
    for p in val_parts:
        gs = torch.load(p, map_location="cpu")
        sanitize_feature_dims_with_schema(gs, rel_F)
        val_graphs.extend(gs)
        del gs
        gc.collect()
    val_loader = PYGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    # ---------------- vocab load & checks ----------------
    vocab_dir = Path(VOCAB_DIR)
    atom_vocab_path = vocab_dir / "atom_vocab.json"
    bond_vocab_path = vocab_dir / "bond_vocab.json"
    assert atom_vocab_path.exists(), f"Missing {atom_vocab_path}. Build vocab first."
    assert bond_vocab_path.exists(), f"Missing {bond_vocab_path}. Build vocab first."
    with open(atom_vocab_path) as f: atom_vocab = json.load(f)
    with open(bond_vocab_path) as f: bond_vocab = json.load(f)

    # small vocab checks
    def _vocab_check(vocab, name):
        assert isinstance(vocab, dict) and len(vocab) >= 2, f"{name}_vocab must have at least ignore + one real token."
        assert "__ignore__" in vocab and vocab["__ignore__"] == 0, f"{name}_vocab must reserve 0 as __ignore__."
        assert "<UNK>" in vocab, f"{name}_vocab must contain <UNK>."
        ids = sorted(vocab.values())
        assert ids[0] == 0 and all(isinstance(i, int) for i in ids), f"{name}_vocab ids must be contiguous ints starting at 0."

    _vocab_check(atom_vocab, "atom")
    _vocab_check(bond_vocab, "bond")

    # ---------------- instantiate encoder using a sanitized probe graph ----------------
    # load one small training shard, sanitize to global schema, and use first graph as sample
    probe_graphs = torch.load(train_parts[0], map_location="cpu")
    sanitize_feature_dims_with_schema(probe_graphs, rel_F)
    sample_graph = probe_graphs[0]
    embedder = HSMPEmbedding(sample=sample_graph, hidden=args.hidden, drop=args.drop,
                             L_edge_k3=4, L_node_k3=6,
                             L_edge_k2=4, L_node_k2=6,
                             L_node_k1=6).to(args.device)
    del probe_graphs, sample_graph
    gc.collect()

    n_params = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    task = GroverLikePretrainTask(embedder, atom_vocab, bond_vocab,
                                  n_fg=85,  # as in your pretrain head
                                  w_atom=2.0, w_bond=1.0, w_fg=5.0,
                                  p_atom=args.p_atom, p_bond=args.p_bond).to(args.device)

    opt = AdamW(task.param_groups(args.lr_encoder, args.lr_heads, args.weight_decay))
    task._optimizer = opt

    # ---------------- training loop: stream shards per epoch ----------------
    best_val = float("inf")
    outdir = Path(PT_CKPT_DIR) / f"{args.outdir}"
    outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # shuffle train shard order each epoch
        rng = random.Random(args.seed + epoch)
        shard_order = list(train_parts)
        rng.shuffle(shard_order)

        # aggregate metrics across shards
        agg_loss = 0.0
        agg_a_correct = 0.0
        agg_a_total = 0
        agg_b_correct = 0.0
        agg_b_total = 0
        n_shard_batches = 0

        for shard_path in shard_order:
            graphs = torch.load(shard_path, map_location="cpu")
            sanitize_feature_dims_with_schema(graphs, rel_F)
            train_loader = PYGDataLoader(graphs, batch_size=args.batch_size, shuffle=True)

            tr_loss, tr_aa, tr_na, tr_ba, tr_nb = train_epoch(task, train_loader, args.device, grad_clip=args.grad_clip)

            # accumulate
            agg_loss += tr_loss * len(train_loader)  # approximate by batch count weighting
            agg_a_correct += tr_aa * tr_na
            agg_a_total   += tr_na
            agg_b_correct += tr_ba * tr_nb
            agg_b_total   += tr_nb
            n_shard_batches += len(train_loader)

            # free
            del graphs, train_loader
            gc.collect()

        # epoch-level averages
        epoch_tr_loss = agg_loss / max(1, n_shard_batches)
        epoch_atom_acc = agg_a_correct / max(1, agg_a_total)
        epoch_bond_acc = agg_b_correct / max(1, agg_b_total)

        va_loss, va_aa, va_na, va_ba, va_nb = eval_epoch(task, val_loader, args.device)

        print(f"[{epoch:02d}] train loss={epoch_tr_loss:.4f} | atom@1={epoch_atom_acc:.3f} ({agg_a_total}) | bond@1={epoch_bond_acc:.3f} ({agg_b_total})")
        print(f"      val   loss={va_loss:.4f} | atom@1={va_aa:.3f} ({va_na}) | bond@1={va_ba:.3f} ({va_nb})")

        meta = {
            "num_training_chunks": len(train_parts),
            "val_chunk_ids": val_ids,
            "hidden": args.hidden,
            "drop": args.drop,
            "atom_vocab_size": 1 + max(atom_vocab.values()),
            "bond_vocab_size": 1 + max(bond_vocab.values()),
            "p_atom": args.p_atom, "p_bond": args.p_bond,
            "task": "GROVER-style ctx pretrain on x_node[1] & ea_f1; ignore_index=0; sparse targets",
            "encoder": "HSMPEmbedding/PolyHeteroModel",
            "seed": args.seed,
            "vocab_dir": str(Path(VOCAB_DIR).resolve()),
            "schema_path": str(Path(SCHEMA_PATH).resolve()),
        }
        torch.save({"encoder": task.embedder.state_dict(), "meta": meta}, str(outdir / "encoder_pretrained.pt"))

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"encoder": task.embedder.state_dict(), "meta": meta}, str(outdir / "encoder_pretrained.best.pt"))
            print("      ✅ saved best → encoder_pretrained.best.pt")


if __name__ == "__main__":
    main()
