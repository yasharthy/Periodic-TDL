#!/usr/bin/env python
# 5_downstream.py  (snapshot-ensembling enabled)
import argparse, json, random, pickle, math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader as PYGDataLoader

from models.embedding import HSMPEmbedding
from models.finetune import HSMPFineTune
from utils.helpers import sanitize_feature_dims, load_dataset
from sklearn.model_selection import train_test_split

import torch_geometric

# ------------------------------ Paths ------------------------------
HETERO_DIR   = "data/HeteroGraphsDownstream"
PROCESSED_DIR = "../data/processed"
PT_CKPT_DIR   = "ckpt_pretrain"

# ------------------------------ Small helpers (unchanged) ------------------------------
def mae(a,b): return (a-b).abs().mean().item()
def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def r2(a,b):
    ss_res = torch.sum((a-b)**2)
    ss_tot = torch.sum((a - torch.mean(a))**2)
    return float(1 - ss_res / (ss_tot + 1e-12))

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    ys, ps = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        loss = loss_fn(pred, y.float())
        total_loss += loss.item() * y.shape[0]
        n += y.shape[0]
        ys.append(y.detach())
        ps.append(pred.detach())
    y_all = torch.cat(ys, dim=0)
    p_all = torch.cat(ps, dim=0)
    metrics = dict(loss=total_loss / max(n,1))
    metrics["rmse"] = rmse(y_all, p_all)
    metrics["mae"]  = mae(y_all, p_all)
    metrics["r2"]   = r2(y_all, p_all)
    return metrics

def train_one_epoch(model, loader, device, loss_fn, opt, grad_clip: float):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        loss = loss_fn(pred, y.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += loss.item() * y.shape[0]
        n += y.shape[0]
    return total_loss / max(n,1)

def print_run_info(args, tr_idx, va_idx, te_idx, hidden, drop):
    print("================================")
    print(f"[env] torch={torch.__version__} | pyg={torch_geometric.__version__} | cuda={torch.version.cuda} "
          f"| device={args.device if torch.cuda.is_available() else 'cpu'}")
    print(f"[ckpt] path={args.ckpt}")
    print(f"[model] hidden={hidden} | out_dim={args.out_dim} | emb_drop={drop} | reg_drop={args.reg_drop}"
          f"| freeze_encoder={args.freeze_encoder}")
    print(f"[train] batch_size={args.batch_size} | epochs={args.epochs} "
          f"| lr_encoder={args.lr_encoder} | lr_head={args.lr_head} "
          f"| weight_decay={args.weight_decay} | grad_clip={args.grad_clip}")
    # === NEW:
    print(f"[warmup] warmup_epochs={args.warmup_epochs} | lr_head_warmup={args.lr_head_warmup}")
    print(f"[restarts] T0={args.T0} T_mult={args.T_mult} eta_min={args.eta_min} | ensemble_last_k={args.ensemble_last_k}")
    print(f"[out] outdir={args.outdir}")
    print("================================\n")
    print(f"[split sizes] train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

# ------------------------------ Dataset I/O (FIXED) ------------------------------
def load_folds_and_csv_smiles(processed_dir: Path, datafile: str, split: int):
    """
    Read fold row indices from {datafile}_folds.pkl (list of dicts),
    map them to SMILES via {datafile}_cleaned.csv.
    We do NOT read labels here.
    """
    import pandas as pd
    csv_path   = processed_dir / f"{datafile}_cleaned.csv"
    folds_path = processed_dir / f"{datafile}_folds.pkl"

    df = pd.read_csv(csv_path)
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)            # e.g., folds[split] -> {"train": [...], "test": [...]}

    if not (0 <= split < len(folds)):
        raise ValueError(f"split must be in [0..{len(folds)-1}]")

    this = folds[split]
    train_rows = df.loc[this["train"]].reset_index(drop=True)
    test_rows  = df.loc[this["test"]].reset_index(drop=True)

    # Return SMILES lists; labels come from the PyG dataset graphs (.y)
    return train_rows["smiles"].tolist(), test_rows["smiles"].tolist()


def align_splits_to_dataset(train_smiles, test_smiles, dataset):
    """
    Map CSV-derived SMILES to indices in the already-featurized PyG dataset.
    """
    sm2idx = {getattr(g, "smiles"): i for i, g in enumerate(dataset)}
    train_idx = [sm2idx[s] for s in train_smiles if s in sm2idx]
    test_idx  = [sm2idx[s] for s in test_smiles  if s in sm2idx]

    # sanity check: ensure labels exist on selected graphs
    for i in train_idx + test_idx:
        if not hasattr(dataset[i], "y"):
            raise RuntimeError(f"Graph at idx={i} (smiles={getattr(dataset[i], 'smiles', None)}) has no .y.")
    return train_idx, test_idx

# ------------------------------ Cycle helper (NEW) ------------------------------
def make_cycle_boundaries(T0: int, T_mult: int, total_epochs: int):
    """Return list of (start_epoch, end_epoch) 1-based inclusive for cosine-warm-restart cycles."""
    cycles, start, Ti, rem = [], 1, max(1, T0), total_epochs
    while rem > 0:
        length = min(Ti, rem)
        cycles.append((start, start + length - 1))
        start += length
        rem -= length
        Ti *= max(T_mult, 1)
    return cycles

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", required=True, help="Basename used for data/HeteroGraphsDownstream/{datafile}_hetero.pt")
    ap.add_argument("--split", type=int, required=True, help="Fold index 0..K-1 (from processed folds.pkl)")
    ap.add_argument("--ckpt", required=True, help="Pretrained encoder bundle path under ckpt_pretrain (contains 'encoder','meta').")

    # model / train hyperparams (unchanged defaults)
    ap.add_argument("--hidden", type=int, default=None)   # if None, take from ckpt meta
    ap.add_argument("--out_dim", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)

    # optimizer
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head",    type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--emb_drop", type=float, default=None)   # if None, take from ckpt meta
    ap.add_argument("--reg_drop", type=float, default=0.0)
    ap.add_argument("--freeze_encoder", action="store_true")

    ap.add_argument("--outdir", default="ckpt_finetune", help="Output dir for finetuned checkpoints")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # === NEW: snapshot ensembling controls
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Stage-A head-only epochs")
    ap.add_argument("--lr_head_warmup", type=float, default=3e-4, help="LR for head-only warmup")
    ap.add_argument("--T0", type=int, default=10, help="First cycle length for cosine restarts")
    ap.add_argument("--T_mult", type=int, default=1, help="Cycle length multiplier")
    ap.add_argument("--eta_min", type=float, default=1e-6, help="Min LR for CAWR")
    ap.add_argument("--ensemble_last_k", type=int, default=5, help="Use last-K cycle snapshots for test ensemble")

    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    # ---------------- Data ----------------
    pt = Path(HETERO_DIR) / f"{args.datafile}_hetero.pt"
    dataset = load_dataset(pt)
    sanitize_feature_dims(dataset)

    train_smiles, test_smiles = load_folds_and_csv_smiles(Path(PROCESSED_DIR), args.datafile, args.split)
    tr_idx_full, te_idx = align_splits_to_dataset(train_smiles, test_smiles, dataset)

    # Consistent val split from the fold's TRAIN partition
    tr_idx, va_idx = train_test_split(
        tr_idx_full, test_size=args.val_frac, shuffle=True, random_state=42, stratify=None
    )

    train_loader = PYGDataLoader([dataset[i] for i in tr_idx], batch_size=args.batch_size, shuffle=True)
    val_loader   = PYGDataLoader([dataset[i] for i in va_idx], batch_size=args.batch_size, shuffle=False)
    test_loader  = PYGDataLoader([dataset[i] for i in te_idx], batch_size=args.batch_size, shuffle=False)

    # ---------------- Model ----------------
    ckpt = torch.load(Path(PT_CKPT_DIR) / f"{args.ckpt}", map_location="cpu")
    meta = ckpt.get("meta", {})
    hidden = args.hidden if args.hidden is not None else int(meta.get("hidden", 768))
    drop   = args.emb_drop if args.emb_drop is not None else float(meta.get("drop", 0.1))

    embedder = HSMPEmbedding(sample=dataset[tr_idx[0]], hidden=hidden, drop=drop,
                             L_edge_k3=4, L_node_k3=6,
                             L_edge_k2=4, L_node_k2=6,
                             L_node_k1=6, num_heads=args.heads).to(args.device)
    model = HSMPFineTune(embedder=embedder, out_dim=args.out_dim, hidden=hidden, drop=args.reg_drop).to(args.device)

    # # Load pretrained encoder weights
    # # Comment this if no need for pretrained weights
    if "encoder" in ckpt:
        model.embedder.load_state_dict(ckpt["encoder"], strict=False)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Print config ----------------
    print_run_info(args, tr_idx, va_idx, te_idx, hidden, drop)

    # ---------------- Stage A: HEAD-ONLY warm-up (NEW) ----------------
    # Freeze encoder
    for p in model.embedder.parameters():
        p.requires_grad = False

    # Use only head params in optimizer
    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optA = AdamW(head_params, lr=args.lr_head_warmup, weight_decay=args.weight_decay)
    schA = CosineAnnealingLR(optA, T_max=max(1, args.warmup_epochs), eta_min=max(1e-8, args.lr_head_warmup*0.1))
    loss_fn = nn.MSELoss(reduction="mean")

    best_overall = {"rmse": float("inf"), "state": None, "epoch": -1, "cycle": 0, "where": "A"}

    if args.warmup_epochs > 0:
        print(f"[Stage A] head-only warmup for {args.warmup_epochs} epochs")
        for ep in range(1, args.warmup_epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, args.device, loss_fn, optA, grad_clip=args.grad_clip)
            schA.step()
            va = evaluate(model, val_loader, args.device, loss_fn)
            print(f" A| epoch {ep:03d}  tr_loss={tr_loss:.6f}  val_rmse={va['rmse']:.4f}  val_r2={va['r2']:.4f}")

            if va["rmse"] < best_overall["rmse"]:
                best_overall = {
                    "rmse": va["rmse"],
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": ep,
                    "cycle": 0,
                    "where": "A",
                }

    # ---------------- Stage B: UNFREEZE + CAWR snapshots (NEW) ----------------
    for p in model.embedder.parameters():
        p.requires_grad = True

    # Your model exposes a param_groups() helper – keep using it for consistency.
    optB = AdamW(model.param_groups(args.lr_encoder, args.lr_head, args.weight_decay, freeze_encoder=False))
    # No momentum/state reset across restarts (default behavior).
    schB = CosineAnnealingWarmRestarts(optB, T_0=args.T0, T_mult=args.T_mult, eta_min=args.eta_min)

    # Cycle boundaries (1-based, inclusive)
    cycles = make_cycle_boundaries(args.T0, args.T_mult, args.epochs)
    print(f"[Stage B] cosine restarts cycles: {cycles}")

    snapshots = []
    global_epoch = 0
    for ci, (start_e, end_e) in enumerate(cycles, start=1):
        best_cycle = {"rmse": float("inf"), "state": None, "epoch": -1}

        for ep in range(start_e, end_e + 1):
            global_epoch += 1
            tr_loss = train_one_epoch(model, train_loader, args.device, loss_fn, optB, grad_clip=args.grad_clip)
            # step scheduler once per epoch (common pattern for CAWR)
            schB.step(global_epoch - 1)

            tr = evaluate(model, train_loader, args.device, loss_fn)
            va = evaluate(model, val_loader, args.device, loss_fn)
            print(f" B| cyc {ci:02d} ep {ep:03d}  tr_rmse={tr['rmse']:.4f} r2={tr['r2']:.4f} | "
                  f"val_rmse={va['rmse']:.4f} r2={va['r2']:.4f}")

            if va["rmse"] < best_cycle["rmse"]:
                best_cycle["rmse"]  = va["rmse"]
                # store CPU copy to avoid CUDA tensors in checkpoint
                best_cycle["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_cycle["epoch"] = ep

            if va["rmse"] < best_overall["rmse"]:
                best_overall = {
                    "rmse": va["rmse"],
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": ep,
                    "cycle": ci,
                    "where": "B",
                }

        # End of cycle → persist snapshot for this cycle
        snap_path = outdir / f"snapshot_cycle{ci}_best.pt"
        # Save bundle with encoder too
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(best_cycle["state"], strict=True)
        torch.save({
            "model": model.state_dict(),
            "encoder": model.embedder.state_dict(),
            "meta": {"hidden": hidden, "out_dim": args.out_dim, "ckpt_init": str(Path(args.ckpt).resolve())}
        }, snap_path)
        # Restore current state (optional)
        model.load_state_dict(cpu_state, strict=False)
        snapshots.append(snap_path)
        print(f"   💾 saved snapshot (cycle {ci}) → {snap_path.name}  @best_val_rmse={best_cycle['rmse']:.4f} ep={best_cycle['epoch']}")

    # ---------------- Final single-model test (last state) ----------------
    if best_overall["state"] is not None:
        model.load_state_dict(best_overall["state"], strict=True)
        print(f"[INFO] Loaded best-overall val model (where={best_overall['where']}, "
            f"cycle={best_overall['cycle']}, epoch={best_overall['epoch']}, "
            f"val_rmse={best_overall['rmse']:.4f}) for single TEST.")

    te_single = evaluate(model, test_loader, args.device, loss_fn)
    print(f"[TEST single] loss={te_single['loss']:.4f} rmse={te_single['rmse']:.4f} "
          f"mae={te_single['mae']:.4f} r2={te_single['r2']:.4f}")

    # ---------------- Snapshot ensemble of last K cycles (NEW) ----------------
    @torch.no_grad()
    def ensemble_predict(paths, make_model, loader, device):
        preds, y_ref = [], None
        for pth in paths:
            # AFTER (robust even when no GPU is available)
            bundle = torch.load(pth, map_location="cpu")
            m = make_model().to(device)
            m.load_state_dict(bundle["model"], strict=False)
            m.eval()
            cur, ys = [], []
            for batch in loader:
                batch = batch.to(device)
                out = m(batch)
                y = batch.y
                if y.dim() == 1:
                    y = y.unsqueeze(-1)
                cur.append(out.detach())
                ys.append(y.detach())
            preds.append(torch.cat(cur, dim=0))
            if y_ref is None:
                y_ref = torch.cat(ys, dim=0)
        mean_pred = torch.stack(preds, dim=0).mean(dim=0)
        return dict(rmse=rmse(y_ref, mean_pred), mae=mae(y_ref, mean_pred), r2=r2(y_ref, mean_pred))

    last_k = min(args.ensemble_last_k, len(snapshots))
    chosen = snapshots[-last_k:]

    def base_model_ctor():
        emb = HSMPEmbedding(sample=dataset[tr_idx[0]], hidden=hidden, drop=drop,
                            L_edge_k3=4, L_node_k3=6,
                            L_edge_k2=4, L_node_k2=6,
                            L_node_k1=6, num_heads=args.heads)
        return HSMPFineTune(embedder=emb, out_dim=args.out_dim, hidden=hidden, drop=args.reg_drop)

    te_ens = ensemble_predict(chosen, base_model_ctor, test_loader, args.device)
    print(f"[TEST ensemble K={last_k}] rmse={te_ens['rmse']:.4f}  mae={te_ens['mae']:.4f}  r2={te_ens['r2']:.4f}")

    # Save last and best-single for convenience (kept from your original behavior)
    torch.save({
        "model": model.state_dict(),
        "encoder": model.embedder.state_dict(),
        "meta": {"hidden": hidden, "out_dim": args.out_dim, "ckpt_init": str(Path(args.ckpt).resolve())}
    }, outdir / "finetuned.last.pt")
    print("   💾 saved last → finetuned.last.pt")

if __name__ == "__main__":
    main()
