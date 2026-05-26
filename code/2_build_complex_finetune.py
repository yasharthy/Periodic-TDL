# 2_build_complex.py
# Build multilevel complexes → convert to PyG HeteroData → save a single
# torch file containing a dict: { SMILES: HeteroData }.
# Minimal-change pattern: read periodic coords, iterate records (now in parallel),
# and write one artifact at data/HeteroGraphs/{DATAFILE}_hetero.pt.

import os
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from rdkit import Chem
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count

# Adjust these imports to your actual module locations
from utils.graphs.multilevel import EpsConfig, build_multilevel_complex
from utils.packing.hetero_convert import to_heterodata_from_complex
from utils.labels.motifs import rdkit_fg_bits_from_smiles

OUTDIR_DEFAULT = "data/HeteroGraphsDownstream"
COORDS_DIR_DEFAULT = "data/Coordinates"


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
    in parallel, converts each to PyG HeteroData, and saves a single torch file
    containing a dict: { SMILES: HeteroData }.

    Returns: output path of the saved .pt file.
    """
    coords_dir = COORDS_DIR_DEFAULT
    outdir = OUTDIR_DEFAULT

    datafile = args.datafile
    procs =args.procs

    in_csv = os.path.join("../data/processed", f"{datafile}_cleaned.csv")
    df = pd.read_csv(in_csv)  # use the same path/df you already rely on
    assert "smiles" in df.columns and "value" in df.columns, "CSV must have 'smiles' and 'value'."
    prop_map = dict(zip(df["smiles"], df["value"]))

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

    if procs is None or procs <= 0:
        procs = max(1, cpu_count() - 1)

    cpx_list = []
    n_total = len(items)
    n_ok = 0

    with Pool(processes=procs) as pool:
        for each in tqdm(pool.imap_unordered(_worker_build_one, items, chunksize=1),
                        total=n_total, desc=f"Building hetero graphs ({datafile})"):
            if each is None:
                continue
            cpx_list.append(each)
            n_ok += 1

    data_list = []
    for smi, cpx in cpx_list:
        hdata = to_heterodata_from_complex(cpx, 
                                           include_atom_ctx=False, 
                                           include_bond_ctx=False)
        hdata.smiles = smi                     # attach identifier

        assert smi in prop_map, f"Missing 'value' for SMILES: {smi}"
        y = float(prop_map[smi])
        hdata.y = torch.tensor([y], dtype=torch.float32)     # [1]

        data_list.append(hdata)

    torch.save(data_list, out_path)   # <— list of HeteroData
    print(f"[OK] Saved {n_ok}/{n_total} hetero graphs → {out_path}")

    return out_path

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", required=True, type=str, help="Dataset basename (e.g., Egc, Tg, Eib)")
    ap.add_argument("--procs", type=int, default=2, help="#workers (0 → use CPU-1)")
    args = ap.parse_args()

    build_and_save_hetero_bundle(args)
