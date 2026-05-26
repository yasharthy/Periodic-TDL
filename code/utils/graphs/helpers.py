import numpy as np
from rdkit import Chem
from utils.geometry.frc import frc_from_min_dist_gudhi_dict


def compute_index_maps(mol_closed, ref_atoms, oid_list):
    # Note: oid2v ignores hydrogens (atoms with None oid)
    # Note: vtx2mol only maps vertices corresponding to heavy atoms
    mol2oid = {i: int(oid) for i, (_, oid) in enumerate(oid_list)}
    oid2v = {int(oid): i for i, (_, _, oid) in enumerate(ref_atoms) if oid is not None}
    vtx2mol = {oid2v[mol2oid[i]]: i for i in range(mol_closed.GetNumAtoms())}
    mol2vtx = {v: k for k, v in vtx2mol.items()}
    return mol2oid, oid2v, vtx2mol, mol2vtx


def compute_curvatures(min_dist, eps_list):
    return {
        float(eps): frc_from_min_dist_gudhi_dict(min_dist, float(eps))
        for eps in eps_list
    }


# --- Minimal helper: get edges/triangles for a cutoff ---
def edges_tris_for_cutoff(cutoff: float, curv_by_eps: dict, min_dist: np.ndarray):
    c = float(cutoff)
    if c in curv_by_eps:  # reuse from curvature run
        _node_curv, edge_curv, tri_curv = curv_by_eps[c]
        edges = sorted(tuple(k) for k in edge_curv.keys())
        tris  = sorted(tuple(k) for k in tri_curv.keys())
        return edges, tris
    
    # else build exactly one Rips complex from min-dist using helper    
    else:    
        rips = gudhi.RipsComplex(distance_matrix=min_dist, max_edge_length=float(cutoff))
        st = rips.create_simplex_tree(max_dimension=3)

        edges = [tuple(sorted(s)) for s, _ in st.get_skeleton(2) if len(s) == 2]
        tris  = [tuple(sorted(s)) for s, _ in st.get_skeleton(3) if len(s) == 3]
        return edges, tris


def apply_sigmoid(arr: np.ndarray, temp: float = 1.0, center: bool = False) -> np.ndarray:
    """
    Elementwise logistic transform with special handling:
      If x < -999 → y = -1
      Else y = 1 / (1 + exp(-x/temp))        in (0,1)
      If center=True, map to (-1,1): y = 2y - 1
    """
    # Create output array
    y = np.empty_like(arr, dtype=np.float32)
    # Mask for "special case"
    mask = arr < -999    
    # Apply normal sigmoid for others
    normal_vals = 1.0 / (1.0 + np.exp(-arr[~mask] / temp))
    if center:
        normal_vals = 2.0 * normal_vals - 1.0
    
    # Assign values
    y[mask] = -1.0
    y[~mask] = normal_vals
    return y


# --- add: tiny helper ---
def cross_maps(src,dst):
    # Node mapping: identity across levels (check same vertex sets)
    n=np.array(sorted(src["node_graph"].nodes()),np.int64)
    m=np.array(sorted(dst["node_graph"].nodes()),np.int64)
    assert (n==m).all(),"[cross node] vertex sets differ"    
    node_ei=np.stack([n,m])

    # Edge mapping: match edge-node ids using stable feat_key=(i,j)
    sk={tuple(d["feat_key"]):i for i,d in src["edge_graph"].nodes(data=True)}
    dk={tuple(d["feat_key"]):i for i,d in dst["edge_graph"].nodes(data=True)}

    # Build pairs for common feat_keys
    pairs=[[sk[k],dk[k]] for k in sk if k in dk]
    edge_ei=np.array(pairs,np.int64).T if pairs else np.zeros((2,0),np.int64)

    return {"node":node_ei,"edge":edge_ei}
