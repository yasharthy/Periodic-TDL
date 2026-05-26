"""
Generating augmented molecules with their coordinates
"""
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Conformer
import numpy as np
import pandas as pd
from collections import Counter  # add once (top of file)

def add_hydrogens(mol):
    """
    Add hydrogens to a molecule that already has 'new_idx' tags
    for heavy atoms. Hydrogens will not get 'new_idx' tags.
    """
    molH = Chem.AddHs(Chem.Mol(mol))  # copy + add hydrogens

    # Tag each atom with 'new_idx'
    for i, atom in enumerate(molH.GetAtoms()):
        atom.SetIntProp("new_idx", i)

    Chem.SanitizeMol(molH)
    return molH

def connect_frags_dimer(monomer):
    """
    Given a monomer with 'new_idx' tags on each atom, duplicate it
    and form a dimer by connecting the 2 dummy atoms (which must exist).
    We do NOT offset any 'new_idx' in the second monomer, so both halves
    end up with the same values for 'new_idx'.
    """

    # 1) Identify the dummy atoms in the (single) monomer
    dum_atms_monomer = [atom for atom in monomer.GetAtoms() if atom.GetSymbol() == '*']
    if len(dum_atms_monomer) != 2:
        raise ValueError("Monomer must have at least two '*' dummy atoms to form a dimer.")

    # We'll connect dum_atms_monomer[1] with dum_atms_monomer[0].
    # You can adjust indexing logic if needed.
    atom1 = dum_atms_monomer[1]
    atom2 = dum_atms_monomer[0]

    # 2) Make a second copy of the monomer (same 'new_idx' tags!)
    monomer2 = Chem.Mol(monomer)

    # 3) Combine the two monomers
    combined = Chem.CombineMols(monomer, monomer2)
    emol = Chem.EditableMol(combined)

    # 4) Figure out the neighbor & dummy indices
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()

    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()

    # The second monomer’s atoms (including dummy) are shifted by monomer.GetNumAtoms()
    shift = monomer.GetNumAtoms()

    # 5) Add the bond between the two neighbor atoms
    bond_order = atom2.GetBonds()[0].GetBondType()  # e.g., single or double
    emol.AddBond(neighbor1_idx, neighbor2_idx + shift, order=bond_order)

    # 6) Remove the two dummy atoms
    emol.RemoveAtom(atom2_idx + shift)  # remove dummy from second monomer
    emol.RemoveAtom(atom1_idx)         # remove dummy from first monomer

    # 7) Final dimer molecule
    dimer = emol.GetMol()
    Chem.SanitizeMol(dimer)  # sanitize the new structure
    return dimer

def remove_orig_idx_from_dummy_atoms(mol):
    """
    Removes the 'new_idx' property from all dummy atoms (atomic number 0).
    Operates in-place and returns the modified molecule.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIntProp("new_idx", -1)
    return mol


def generate_augmented_mols(monomer):
    dimer = connect_frags_dimer(monomer)
    dummies = [i for i, a in enumerate(dimer.GetAtoms()) if a.GetSymbol() == '*']

    # atoms along the shortest dummy-to-dummy path (exclude the dummies)
    path = list(Chem.rdmolops.GetShortestPath(dimer, dummies[0], dummies[1]))[1:-1]
    # bonds along that path, in order
    path_bonds = [dimer.GetBondBetweenAtoms(a, b).GetIdx() for a, b in zip(path, path[1:])]
    # keep only acyclic bonds, preserving order along the path
    corridor = [bid for bid in path_bonds if not dimer.GetBondWithIdx(bid).IsInRing()]
    if len(corridor) < 2:
        return [monomer]

    # Pair across the center (no ends-pairing)
    if len(corridor) % 2:
        del corridor[len(corridor) // 2]
    mid = len(corridor) // 2
    left, right = corridor[:mid], corridor[mid:]
    bond_pairs = list(zip(left, right))

    pos = {a: i for i, a in enumerate(path)}

    # --- BASELINE COMPOSITION (includes '*' and any explicit Hs) ---
    base_comp = Counter(a.GetSymbol() for a in monomer.GetAtoms())

    new_ru_list = []
    for bL, bR in bond_pairs:
        # inner atom = endpoint closer to the corridor center
        BL = dimer.GetBondWithIdx(bL); aL1, aL2 = BL.GetBeginAtomIdx(), BL.GetEndAtomIdx()
        innerL = aL1 if pos[aL1] > pos[aL2] else aL2
        BR = dimer.GetBondWithIdx(bR); aR1, aR2 = BR.GetBeginAtomIdx(), BR.GetEndAtomIdx()
        innerR = aR1 if pos[aR1] < pos[aR2] else aR2

        cut = Chem.FragmentOnBonds(dimer, [bL, bR], addDummies=True, dummyLabels=[(0, 0), (0, 0)])

        frag_sets = Chem.GetMolFrags(cut, asMols=False, sanitizeFrags=True)
        idx_mid = next((i for i, atoms in enumerate(frag_sets) if innerL in atoms and innerR in atoms), None)
        if idx_mid is None:
            continue
        frags = list(Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True))
        # print("[AUG Probe] num atoms in fragments", [f.GetNumAtoms() for f in frags])

        mid_ru = frags[idx_mid]

        # --- COMPOSITION ASSERTION (foolproof & tiny) ---
        aug_comp = Counter(a.GetSymbol() for a in mid_ru.GetAtoms())
        assert aug_comp == base_comp, (
            f"Augmentation composition mismatch.\n"
            f"Missing from aug: {base_comp - aug_comp}\n"
            f"Extra in aug:     {aug_comp - base_comp}"
        )
        assert monomer.GetNumAtoms() == mid_ru.GetNumAtoms(), ("Number of atoms mismatch across augmentations!")

        new_ru_list.append(mid_ru)

    augmented_reps = [monomer] + new_ru_list

    unique_mols, seen = [], set()
    for mol in augmented_reps:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Skipping unsanitizable molecule due to error: {e}")
            continue
        mol = remove_orig_idx_from_dummy_atoms(mol)
        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi not in seen:
            seen.add(smi)
            unique_mols.append(mol)

    return unique_mols

# Incorrect code
# def generate_augmented_mols(monomer):
#     dimer = connect_frags_dimer(monomer)
#     dum_atm_idx_dimer = [idx for idx, atom in enumerate(dimer.GetAtoms()) if atom.GetSymbol() == '*']

#     sp_bw_dummy_atoms = Chem.rdmolops.GetShortestPath(dimer, dum_atm_idx_dimer[0], dum_atm_idx_dimer[1])[1:-1]
#     sp_bonds = [dimer.GetBondBetweenAtoms(first, second).GetIdx()
#                 for first, second in zip(sp_bw_dummy_atoms, sp_bw_dummy_atoms[1:])]
#     sp_nonring_bonds = [idx for idx in sp_bonds if not dimer.GetBonds()[idx].IsInRing()]
#     del sp_nonring_bonds[(len(sp_nonring_bonds)-1)//2]
#     bond_pair_list = np.reshape(sp_nonring_bonds, (2, -1))

#     new_ru_list = []
#     for bond_pair in bond_pair_list.T:
#         bond_pair = [val.item() for val in list(bond_pair)]
#         mol_f = Chem.FragmentOnBonds(dimer, bond_pair, addDummies=True, dummyLabels=[(0, 0), (0, 0)])
#         mol_f1 = list(Chem.rdmolops.GetMolFrags(mol_f, asMols=True))
#         new_ru_list.append(mol_f1[1])

#     augmented_reps = [monomer] + new_ru_list

#     unique_mols = []
#     seen_smiles = set()

#     for mol in augmented_reps:
#         try:
#             Chem.SanitizeMol(mol)  # sanitize each mol
#         except Exception as e:
#             print(f"Skipping unsanitizable molecule due to error: {e}")
#             continue

#         mol = remove_orig_idx_from_dummy_atoms(mol)
#         smiles = Chem.MolToSmiles(mol, canonical=True)

#         if smiles not in seen_smiles:
#             seen_smiles.add(smiles)
#             unique_mols.append(mol)

#     return unique_mols


"""
Get optimized 3d structure of a monomer
"""

def replace_dummies_by_bond_type(mol):
    rwmol = Chem.RWMol(mol)
    dummy_atoms = [(a.GetIdx(), a.GetBonds()[0]) for a in rwmol.GetAtoms() if a.GetAtomicNum() == 0]

    type_to_atomic_num = {
        Chem.BondType.SINGLE: 17,  # Cl
        Chem.BondType.DOUBLE: 8,   # O
        Chem.BondType.TRIPLE: 7    # N
    }

    for idx, bond in sorted(dummy_atoms, reverse=True):
        neighbor_idx = bond.GetOtherAtomIdx(idx)
        bond_type = bond.GetBondType()
        new_atom_idx = rwmol.AddAtom(Chem.Atom(type_to_atomic_num[bond_type]))
        rwmol.GetAtomWithIdx(new_atom_idx).SetIntProp("new_idx", -1)
        rwmol.AddBond(neighbor_idx, new_atom_idx, bond_type)
        rwmol.RemoveBond(idx, neighbor_idx)
        rwmol.RemoveAtom(idx)

    Chem.SanitizeMol(rwmol)
    return rwmol.GetMol()

def get_dummy_and_neighbor_indices(mol):
    dummy_and_neighbors = []

    for atom in mol.GetAtoms():
        if atom.HasProp("new_idx") and atom.GetIntProp("new_idx") == -1:
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                neighbor_idx = neighbors[0].GetIdx()
                dummy_and_neighbors.append((atom.GetIdx(), neighbor_idx))
            else:
                raise ValueError(f"Dummy atom {atom.GetIdx()} has {len(neighbors)} neighbors; expected 1.")

    if len(dummy_and_neighbors) != 2:
        raise ValueError(f"Expected 2 dummy atoms, found {len(dummy_and_neighbors)}.")

    # Unpack as dum1, dum2, neigh1, neigh2
    (dum1, neigh1), (dum2, neigh2) = dummy_and_neighbors
    return dum1, dum2, neigh1, neigh2

def find_best_conf_from_mol(mol, dum1, dum2, atom1, atom2):
    # # Make a copy of the input mol and add hydrogens
    # mol = Chem.AddHs(Chem.Mol(mol))

    # Generate multiple conformers
    params = AllChem.ETKDGv3()
    params.numThreads = 8  # Use all available cores
    params.pruneRmsThresh = 0.5  # Adjust as needed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=30, params=params)

    # Store cid, dihedral angle, and energy
    cid_list = []
    for cid in cids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid)
        conf = mol.GetConformer(cid)
        ffu = AllChem.UFFGetMoleculeForceField(mol, confId=cid)

        dihedral_angle = abs(
            rdMolTransforms.GetDihedralDeg(conf, int(dum1), int(atom1), int(atom2), int(dum2))
        )

        cid_list.append([cid, dihedral_angle, ffu.CalcEnergy()])

    # Filter by dihedral angle and energy
    cid_df = pd.DataFrame(cid_list, columns=['cid', 'Dang', 'Energy'])
    cid_df = cid_df.sort_values(by='Dang', ascending=False)
    top_angle = cid_df.iloc[0]['Dang']
    cid_df = cid_df[cid_df['Dang'] > top_angle - 8.0]
    cid_df = cid_df.sort_values(by='Energy', ascending=True)

    # Use best conformer
    best_cid = int(cid_df.iloc[0]['cid'])
    best_conf = mol.GetConformer(best_cid)
    conf_copy = Conformer(best_conf)
    mol.RemoveAllConformers()
    mol.AddConformer(conf_copy, assignId=True)
    
    return mol

def get_3d_opt_mol(m2): # When conformer generation fails
    # Get 2D coordinates
    # print("Getting 2D coordinates...")
    # Chem.AssignStereochemistry(m2, force=True, tryExplicit=False)
    AllChem.Compute2DCoords(m2)
    # print(a)

    # Make 3D mol
    # print("Generating 3D coordinates...")
    AllChem.EmbedMolecule(m2)

    # Optimize 3D str
    # print("Optimizing 3D coordinates...")
    AllChem.UFFOptimizeMolecule(m2, maxIters=200)
    return m2


"""
Get 3D coordinates and pairwise distances of all augemntations
"""

def get_coordinates_tensor_by_new_idx(mol_list):
    """
    Returns:
    - coords: (n_mols, n_atoms, 3) NumPy array aligned by new_idx.
    - ref_atoms: list of (atom_symbol, new_idx, orig_idx_or_None) in tensor order.
    - canonical_smiles: list of canonical SMILES for each mol in mol_list.
    """
    # Build reference (from first mol): keep atoms with new_idx != -1
    ref_atoms = []
    for atom in mol_list[0].GetAtoms():
        if atom.HasProp("new_idx"):
            nid = atom.GetIntProp("new_idx")
            if nid != -1:
                oid = atom.GetIntProp("orig_idx") if atom.HasProp("orig_idx") else None
                ref_atoms.append((atom.GetSymbol(), nid, oid))

    # Sort by new_idx, fix order, and map new_idx -> tensor index
    ref_atoms.sort(key=lambda x: x[1])  # sort by new_idx
    new_idx_order = [nid for (_, nid, _) in ref_atoms]
    new_idx_to_tensor_idx = {nid: i for i, nid in enumerate(new_idx_order)}

    n_mols = len(mol_list)
    n_atoms = len(new_idx_order)
    coords = np.zeros((n_mols, n_atoms, 3))

    # Fill coordinates for each augmentation
    for mol_id, mol in enumerate(mol_list):
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.HasProp("new_idx"):
                nid = atom.GetIntProp("new_idx")
                if nid != -1 and nid in new_idx_to_tensor_idx:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coords[mol_id, new_idx_to_tensor_idx[nid]] = [pos.x, pos.y, pos.z]

    canonical_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mol_list]

    # has_oid = sum(oid is not None for _,_,oid in ref_atoms)
    # uniq_oid = len({oid for _,_,oid in ref_atoms if oid is not None})
    # print(f"[S2] ref_atoms with oid: {has_oid}/{len(ref_atoms)} | unique_oids={uniq_oid}")

    return coords, ref_atoms, canonical_smiles


def compute_pairwise_distances(coords_tensor):
    """
    Computes pairwise Euclidean distances between atoms
    for each augmentation.
    
    Args:
        coords_tensor: shape (n_mols, n_atoms, 3)
        
    Returns:
        dist_tensor: shape (n_mols, n_atoms, n_atoms)
    """
    n_mols, n_atoms, _ = coords_tensor.shape
    dist_tensor = np.zeros((n_mols, n_atoms, n_atoms))

    for i in range(n_mols):
        diff = coords_tensor[i][:, np.newaxis, :] - coords_tensor[i][np.newaxis, :, :]
        dist_tensor[i] = np.linalg.norm(diff, axis=2)

    return dist_tensor

def compute_min_distance_across_augmentations(dist_tensor):
    min_dist = np.min(dist_tensor, axis=0)  # shape: (n_atoms, n_atoms)
    return min_dist


"""
Wrapper function
"""
from typing import Tuple

def all_coordinates(mol):

    # 1. Generate augmented monomers
    mol = add_hydrogens(mol)
    augmented_mols = generate_augmented_mols(mol)
    capped_augmented_mols = [replace_dummies_by_bond_type(each) for each in augmented_mols]

    # 2. Optimize conformers
    optimized_mols = []
    for mol in capped_augmented_mols:
        try:
            optimized_mols.append(get_3d_opt_mol(mol))
        except Exception as e:
            print(f"Failed to optimize molecule: {e}. Skipping this molecule.")
            continue

    if not optimized_mols:
        raise ValueError("No valid 3D conformers generated.")

    # 3. Get aligned coordinates and atom identities
    coords, ref_atoms, _ = get_coordinates_tensor_by_new_idx(optimized_mols)
    return coords, ref_atoms

def coords_to_dist(coords: np.ndarray) -> np.ndarray:
    """
    Computes the minimum distance matrix across all augmentations.

    Args:
        coords: shape (n_mols, n_atoms, 3)

    Returns:
        min_dist: shape (n_atoms, n_atoms)
    """
    dist_tensor = compute_pairwise_distances(coords)
    min_dist = compute_min_distance_across_augmentations(dist_tensor)
    return dist_tensor, min_dist

def filter_distance_matrix_by_cutoff(
    distance_matrix: np.ndarray,
    cutoff_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters a distance matrix using a lower and upper cutoff range.

    Parameters:
    - distance_matrix: np.ndarray of shape (N, N), real-valued distances
    - cutoff_range: (lower_cutoff, upper_cutoff)

    Returns:
    - drug_dis: binary adjacency matrix (1 if within range, else 0)
    - drug_dis_real: real-valued distances, zeroed out where outside range
    """
    lower, upper = cutoff_range
    N = distance_matrix.shape[0]

    drug_dis = np.ones((N, N), dtype=float)
    drug_dis_real = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            d = distance_matrix[i, j]
            if lower < d < upper:
                drug_dis[i, j] = 1.0
                drug_dis_real[i, j] = d
            else:
                drug_dis[i, j] = 0.0
                drug_dis_real[i, j] = 0.0

    return drug_dis, drug_dis_real
