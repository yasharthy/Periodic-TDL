#!/usr/bin/env python
# predict_pretrain_Tg_with_deploy_split0.py
#
# Loads the *first* deployment model from 6_depolyment_2.py outputs
# (ckpt_deploy/<datafile>/split_0/finetuned.best.pt) and predicts Tg for
# all polymers in your 1M pretrain heterograph shards, writing:
# pSMILES,Tg_pred
#
# Assumes your pretrain heterographs were built as shards:
# data/HeteroGraphsPretrain/<pretrain_dir>/<stem>_hetero_part*.pt
# (each shard is a list[HeteroData] with g.smiles set). :contentReference[oaicite:0]{index=0}
#
# The deployment checkpoint format/path matches 6_depolyment_2.py. 

import argparse
from pathlib import Path
from glob import glob
import csv

import torch
from torch_geometric.loader import DataLoader as PYGDataLoader

from models.embedding import HSMPEmbedding
from models.finetune import HSMPFineTune
from utils.helpers import sanitize_feature_dims


def iter_part_files(root: Path, stem: str):
    files = sorted(glob(str(root / f"{stem}_hetero_part*.pt")))
    if not files:
        raise FileNotFoundError(f"No part files found: {root}/{stem}_hetero_part*.pt")
    return [Path(f) for f in files]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_datafile", required=True,
                    help="Name used during deployment training (folder under deploy_dir), e.g., Tg")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")

    ap.add_argument("--deploy_dir", default="ckpt_deploy",
                    help="Base deployment output dir from 6_depolyment_2.py")
    ap.add_argument("--split_id", type=int, default=0, help="Which fold model to use (default: 0)")

    ap.add_argument("--pretrain_hetero_dir", default="data/HeteroGraphsPretrain/pretrain_1M_v1",
                    help="Directory containing pretrain heterograph shards")
    ap.add_argument("--pretrain_stem", default="pretrain_1M",
                    help="Shard filename stem (before _hetero_partXX.pt)")

    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--emb_drop", type=float, default=0.1,
                    help="Must match deployment emb_drop (not stored in ckpt). Default matches 6_depolyment_2.py")
    ap.add_argument("--reg_drop", type=float, default=0.0,
                    help="Must match deployment reg_drop (default matches 6_depolyment_2.py)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    deploy_ckpt = (
        Path(args.deploy_dir)
        / args.deploy_datafile
        / f"split_{args.split_id}"
        / "finetuned.best.pt"
    )
    if not deploy_ckpt.exists():
        raise FileNotFoundError(f"Missing deployment checkpoint: {deploy_ckpt}")

    ckpt = torch.load(deploy_ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})
    hidden = int(meta.get("hidden", 768))
    out_dim = int(meta.get("out_dim", 1))
    if out_dim != 1:
        raise ValueError(f"Expected out_dim=1 for Tg regression; got out_dim={out_dim}")

    # Probe one graph to instantiate the embedder with correct feature shapes
    pretrain_root = Path(args.pretrain_hetero_dir)
    part_files = iter_part_files(pretrain_root, args.pretrain_stem)
    probe_graphs = torch.load(part_files[0], map_location="cpu")
    sanitize_feature_dims(probe_graphs)
    sample_graph = probe_graphs[0]
    del probe_graphs

    embedder = HSMPEmbedding(
        sample=sample_graph,
        hidden=hidden,
        drop=args.emb_drop,
        L_edge_k3=4, L_node_k3=6,
        L_edge_k2=4, L_node_k2=6,
        L_node_k1=6,
    )
    model = HSMPFineTune(embedder=embedder, out_dim=out_dim, hidden=hidden, drop=args.reg_drop)

    # Load weights
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(args.device)
    model.eval()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pSMILES", "Tg_pred"])

    # Stream shards -> predict -> append rows
    for p in part_files:
        graphs = torch.load(p, map_location="cpu")
        sanitize_feature_dims(graphs)

        loader = PYGDataLoader(graphs, batch_size=args.batch_size, shuffle=False)

        rows = []
        for batch in loader:
            batch = batch.to(args.device)
            pred = model(batch).detach().view(-1).cpu().tolist()

            # smiles should collate into a Python list of strings
            if hasattr(batch, "smiles"):
                smiles = batch.smiles
            else:
                # fallback
                smiles = [str(i) for i in range(len(pred))]

            if len(smiles) != len(pred):
                raise RuntimeError(f"Batch smiles length {len(smiles)} != preds length {len(pred)}")

            rows.extend(zip(smiles, pred))

        # append rows for this shard
        with out_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)

        del graphs, loader, rows
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[OK] Wrote predictions → {out_path}")


if __name__ == "__main__":
    main()
