# utils/grover_like_fg.py
from typing import Tuple, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Fragments as F

_FG_NAMES = [n for n in dir(F) if n.startswith("fr_")]

def rdkit_fg_bits_from_smiles(smiles: str) -> Tuple[np.ndarray, int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Bad SMILES: {smiles}"
    counts = np.array([getattr(F, n)(mol) for n in _FG_NAMES], dtype=float)
    bits = (counts != 0).astype(np.float32)
    return bits, len(_FG_NAMES)

if __name__ == "__main__":
    # smiles = "[*]CCCCCC[N+](C)(C)CCCCC[N+]([*])(C)C"
    smiles = "[*]c1cc2c(s1)-c1sc(-c3cc4c(s3)-c3sc([*])cc3C4(CC(CC)CCCC)CC(CC)CCCC)cc1C2=O"
    smiles = "O=C(Nc1ccc(cc1)C(=O)Nc1ccc(cc1)Oc1ccc(cc1)NC(=O)c1ccc(cc1)N1C(=O)c2c(C1=O)cc(cc2)C(=O)*)Nc1ccc(cc1)Cc1ccc(cc1)NC(=O)Nc1ccc(cc1)C(=O)Nc1ccc(cc1)Oc1ccc(cc1)NC(=O)c1ccc(cc1)N1C(=O)c2c(C1=O)cc(cc2)*"

    # print(_FG_NAMES)
    print(rdkit_fg_bits_from_smiles(smiles))