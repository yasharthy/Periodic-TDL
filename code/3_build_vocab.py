#!/usr/bin/env python
# 3_build_vocab.py
import os, json, argparse
from pathlib import Path
from utils.helpers import load_all_parallel_proc
from utils.labels.vocab import atom_and_bond_vocab

# HETERO_DIR = "data/HeteroGraphs"
HETERO_DIR = "data/HeteroGraphsPretrain/pretrain_1M_v1"
OUT_DIR = "data/CtxVocab/"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", required=True)
    ap.add_argument("--num_chunks", type=int, required=True)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--no_unk", action="store_true") # Don't add this argument if you want to add unk
    args = ap.parse_args()

    graphs = load_all_parallel_proc(HETERO_DIR, args.datafile, args.num_chunks, args.max_workers)
    atom_vocab, bond_vocab = atom_and_bond_vocab(graphs, args.min_freq, add_unk=not args.no_unk)

    os.makedirs(args.outdir, exist_ok=True)
    (Path(args.outdir)/"atom_vocab.json").write_text(json.dumps(atom_vocab, indent=2))
    (Path(args.outdir)/"bond_vocab.json").write_text(json.dumps(bond_vocab, indent=2))

    print(f"Saved vocabs to {args.outdir}: atoms={len(atom_vocab)-1} bonds={len(bond_vocab)-1} (excl ignore)")

if __name__ == "__main__":
    main()

# Old script for loading hetero graph (before the graph list was chunked)
    # hetero_path = os.path.join(HETERO_DIR, f"{args.datafile}_hetero.pt")
    # # print(hetero_path)
    # graphs = _load_list_or_dict(hetero_path)
    # print(graphs[0])
    # graphs = []
    # for ci in range(int(args.num_chunks)):
    #     hetero_path_i = os.path.join(HETERO_DIR, f"{args.datafile}_hetero_part{ci:02d}.pt")
    #     part_graphs = _load_list(hetero_path_i)
    #     print(f"[INFO] Loaded part {ci:02d}: {len(part_graphs)} heterographs")
    #     graphs += part_graphs

    # print(f"[INFO] Total heterographs loaded: {len(graphs)}")
