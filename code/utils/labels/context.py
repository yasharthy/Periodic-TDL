# utils/labels/context.py
"""
Code sourced from GROVER https://github.com/tencent-ailab/grover/blob/main/grover/data/task_labels.py
With one minor modification: for bond we label by BondType, IsConjugated, IsInRing to keep consistent with the input features
"""
from rdkit import Chem
from collections import Counter

# ---- utilities ----
# Exactly like GROVER: include BondType, Stereo, BondDir in this order, stringified.
def _bond_feature_name(bond: Chem.rdchem.Bond) -> str:
    bt = str(bond.GetBondType())     # e.g., 'SINGLE' / 'AROMATIC'
    st = str(bond.GetIsConjugated())       # e.g., 'STEREONONE'
    bd = str(bond.IsInRing())      # e.g., 'NONE'
    return f"({bt}-{st}-{bd})"

def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output


def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + _bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = _bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output

if __name__ == "__main__":
    # smiles = "[*]CCCCCC[N+](C)(C)CCCCC[N+]([*])(C)C"
    smiles = "[*]c1cc2c(s1)-c1sc(-c3cc4c(s3)-c3sc([*])cc3C4(CC(CC)CCCC)CC(CC)CCCC)cc1C2=O"
    mol = Chem.MolFromSmiles(smiles)

    for mol_idx in range(mol.GetNumAtoms()):
        atm = mol.GetAtomWithIdx(mol_idx)
        print(atm.GetSymbol(), atom_to_vocab(mol, atm))
    
    for b in mol.GetBonds():
        print(f"({b.GetBeginAtom().GetSymbol()},{b.GetEndAtom().GetSymbol()})", bond_to_vocab(mol, b))
