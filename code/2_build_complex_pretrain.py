"""
2_build_complex_pretrain.py

Build multilevel complexes and convert to PyG HeteroData objects.
Modified for PRETRAINING ONLY:
- Always includes atom and bond context features (no finetune mode).
- Randomizes the dataset and splits into chunks for saving.
- Number of chunks is controlled by CLI argument (--chunks, default=10).
- Each chunk is saved separately in data/HeteroGraphsPretrain as
  {DATAFILE}_hetero_partXX.pt.
"""
import os
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import random
import psutil

from rdkit import Chem
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count

# Adjust these imports to your actual module locations
from utils.graphs.multilevel import EpsConfig, build_multilevel_complex
from utils.packing.hetero_convert import to_heterodata_from_complex
from utils.labels.motifs import rdkit_fg_bits_from_smiles

# !check thiss
# OUTDIR_DEFAULT = "data/HeteroGraphsPretrain"
# OUTDIR_DEFAULT = "data/HeteroGraphsAcrylate3"
OUTDIR_DEFAULT = "data/HeteroGraphsAcrylateLit"
COORDS_DIR_DEFAULT = "data/Coordinates"

def _mem_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def _worker_build_one(args):
    smi, rec = args
    try:
        mol_closed = Chem.Mol(rec["mol_closed"])
        coords, ref_atoms = rec["coords"], rec["ref_atoms"]
        oid_list = rec["oid_per_closed_idx"]

        eps_cfg = EpsConfig() # Default configuration
        cpx = build_multilevel_complex(mol_closed, coords, ref_atoms, oid_list, eps_cfg)
        return smi, cpx
    except Exception as e:
        print(f"[WARN] {smi}: {e}")
        return None


def build_and_save_hetero_bundle(args) -> str:
    """
    Reads periodic coordinate records for `datafile`, builds multilevel complexes
    in parallel, converts each to PyG HeteroData, and saves a torch files
    containing a dict in chunks: { SMILES: HeteroData }.
    """
    coords_dir = COORDS_DIR_DEFAULT
    outdir = OUTDIR_DEFAULT

    datafile = args.datafile
    procs =args.procs
    n_chunks = args.chunks

    coords_pkl = os.path.join(coords_dir, f"{datafile}_periodic.pkl")
    if not os.path.exists(coords_pkl):
        raise FileNotFoundError(f"Missing coordinates pickle: {coords_pkl}")

    with open(coords_pkl, "rb") as f:
        rows: List[Dict[str, Any]] = pickle.load(f)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{datafile}_hetero.pt")

    # Prepare items as (smiles, rec) pairs
    items: List[Tuple[str, Dict[str, Any]]] = []
    for rec in rows:
        smi = rec.get("smiles")
        if not smi:
            # skip records without a SMILES key
            continue
        items.append((smi, rec))

    # --- deterministic shuffle on indices (resume-friendly) ---
    rng = random.Random(42)
    n = len(items)
    idx = list(range(n))
    rng.shuffle(idx)

    # --- split indices into n_chunks, without copying item payloads ---
    chunk_size = max(1, (n + n_chunks - 1) // n_chunks)
    # idx_slices only hold integers (lightweight), not the heavy records
    idx_slices = [idx[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    total = n
    n_ok_global = 0

    for ci, idx_slice in enumerate(idx_slices):
        # allow resuming: skip if output already exists
        out_path_i = os.path.join(outdir, f"{datafile}_hetero_part{ci:02d}.pt")
        if os.path.exists(out_path_i):
            print(f"[SKIP] {out_path_i} exists; skipping part {ci+1}/{n_chunks}")
            continue

        if not idx_slice:
            # empty slice (happens when n < n_chunks)
            continue

        # materialize only this chunk's (smi, rec) pairs (list of references, not deep copies)
        print(f"[MEM] Before chunk {ci+1}/{n_chunks}: { _mem_gb():.2f} GB")
        chunk = [items[j] for j in idx_slice]
        n_total = len(chunk)
        n_ok = 0

        data_list = []
        try:
            # fresh workers per chunk; maxtasksperchild avoids slow leaks in long runs
            with Pool(processes=procs, maxtasksperchild=50) as pool:
                for each in tqdm(
                    pool.imap_unordered(_worker_build_one, chunk, chunksize=1),
                    total=n_total,
                    desc=f"Building hetero graphs ({datafile}) [part {ci+1}/{n_chunks}]"
                ):
                    if each is None:
                        continue

                    smi, cpx = each
                    # convert on-the-fly (no cpx_list retained)
                    hdata = to_heterodata_from_complex(
                        cpx,
                        include_atom_ctx=True,
                        include_bond_ctx=True
                    )
                    hdata.smiles = smi
                    fg_list, _ = rdkit_fg_bits_from_smiles(smi)
                    hdata.fg = torch.tensor(fg_list, dtype=torch.float32)
                    data_list.append(hdata)
                    n_ok += 1

            # save this part and free memory
            torch.save(data_list, out_path_i)
            print(f"[OK] Saved {n_ok}/{n_total} hetero graphs → {out_path_i}")
            n_ok_global += n_ok

        finally:
            # ensure per-chunk memory is dropped even on exceptions
            print(f"[MEM] After chunk {ci+1}/{n_chunks}:  { _mem_gb():.2f} GB")
            del data_list, chunk
            import gc; gc.collect()
            print(f"[MEM] After using Garbage Collector {ci+1}/{n_chunks}:  { _mem_gb():.2f} GB")

    print(f"[DONE] Saved {n_ok_global}/{total} hetero graphs across pending parts in {outdir}")

    return None

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", required=True, type=str, help="Dataset basename (e.g., Egc, Tg, Eib)")
    ap.add_argument("--procs", type=int, default=2, help="#workers")
    ap.add_argument("--chunks", type=int, default=100,
                    help="Number of chunks to split dataset into (default: 10)")

    args = ap.parse_args()

    build_and_save_hetero_bundle(args)
