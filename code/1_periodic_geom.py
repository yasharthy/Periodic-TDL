# periodic_geom.py  (Script 1, lean version)
# Written By: Yasharth Yadav (with help  from ChatGPT)

import argparse
import os, pickle, logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from multiprocessing import Pool, cpu_count

from utils.chemistry.featurize_RDKit import build_monocycle
from utils.geometry.periodic_repn import all_coordinates

OUT_DIR  = "data/Coordinates"
PKL_NAME = "{datafile}_periodic.pkl"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def _process_smiles(s: str, smiles_col: str = "smiles"):
    """Helper to process one SMILES → record dict, or None on failure."""
    if not s or Chem.MolFromSmiles(s) is None:
        logging.error("Invalid SMILES: %s | MolFromSmiles returned None", s)
        return None
    try:
        mol_closed, mol_open = build_monocycle(s)
        coords, ref_atoms = all_coordinates(mol_open)

        # # Probe to check presence of orig_idx
        # def count_orig(m): 
        #     return sum(a.HasProp("orig_idx") for a in m.GetAtoms()), m.GetNumAtoms()

        # open_has, open_N   = count_orig(mol_open)
        # closed_has, closed_N = count_orig(mol_closed)
        # print(f"[S0] orig_idx — open {open_has}/{open_N} | closed {closed_has}/{closed_N}")

        # build a closed-index -> orig_idx array (None for atoms without it)
        oid_list = [(a.GetSymbol(), a.GetIntProp("orig_idx"))
            for a in mol_closed.GetAtoms()]

        # mapping = _build_mapping(mol_closed, ref_atoms)
        return {
            "smiles": s,
            "mol_closed": mol_closed.ToBinary(),
            "coords": coords,
            "ref_atoms": ref_atoms,
            "oid_per_closed_idx": oid_list
        }
    except Exception as e:
        logging.error("SMILES failed: %s | Error: %s", s, str(e))
        return None
    
def build_periodic_dataset(datafile: str, smiles_col: str = "smiles", n_jobs: int = None):
    os.makedirs(OUT_DIR, exist_ok=True)
    in_csv = os.path.join("../data/processed", f"{datafile}_cleaned.csv")
    out_pkl = os.path.join(OUT_DIR, PKL_NAME.format(datafile=datafile))
    logfile = os.path.join(LOG_DIR, f"{datafile}_failed_smiles.log")
    logging.basicConfig(
        filename=logfile,
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s")

    df = pd.read_csv(in_csv)
    smiles_list = df[smiles_col].astype(str).tolist()

    n_jobs = n_jobs or max(cpu_count() - 1, 1)
    print(f"Using {n_jobs} worker processes")

    rows_out = []
    with Pool(processes=n_jobs) as pool:
        for rec in tqdm(pool.imap_unordered(_process_smiles, smiles_list),
                        total=len(smiles_list), desc="Periodic monocycle geometry"):
            if rec is not None:
                rows_out.append(rec)

    with open(out_pkl, "wb") as f:
        pickle.dump(rows_out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(rows_out)}/{len(smiles_list)} molecules → {out_pkl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datafile", type=str, required=True, help="basename of {datafile}_cleaned.csv in data/processed/")
    ap.add_argument("--smiles_col", type=str, default="smiles")
    args = ap.parse_args()

    build_periodic_dataset(args.datafile, smiles_col=args.smiles_col, n_jobs=64)

    # test_pkl = os.path.join(OUT_DIR, PKL_NAME.format(datafile=args.datafile))
    # with open(test_pkl, "rb") as f:
    #     rows = pickle.load(f)

    # for rec in rows:
    #     mol_closed = Chem.Mol(rec["mol_closed"])
    #     print([a.HasProp("orig_idx") for a in mol_closed.GetAtoms()])

