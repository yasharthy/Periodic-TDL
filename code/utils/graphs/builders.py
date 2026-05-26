import networkx as nx
import numpy as np

from utils.chemistry.featurize_RDKit import atom_fp, bond_fp
from utils.chemistry.featurize_RDKit import AtomConfig, BondConfig
from utils.labels.context import atom_to_vocab, bond_to_vocab

from utils.graphs.helpers import apply_sigmoid, edges_tris_for_cutoff


ATOM_CFG = AtomConfig(
    element_type=True, degree=True, implicit_valence=True,
    formal_charge=True, num_rad_e=True, hybridization=True,
    combo_hybrid=False, aromatic=True)


BOND_CFG = BondConfig(bond_type=True, conjugation=True, ring=True)


def build_closed_graph(mol_closed, mol2vtx, curv_by_eps, eps_range):
    feat_dict = {
        "node": {},      # key: v (Rips/closed vertex id) -> np.ndarray
        "edge": {},      # key: (i,j) sorted              -> np.ndarray
        "tri": {}        # unused here
    }

    # ---------------- Node graph with node pointers ----------------
    Gnode = nx.Graph()
    for mol_idx in range(mol_closed.GetNumAtoms()):
        vtx = mol2vtx[mol_idx]
        atm = mol_closed.GetAtomWithIdx(mol_idx)

        atom_feat = atom_fp(atm, ATOM_CFG)
        atom_context = atom_to_vocab(mol_closed, atm) # added atom context from GROVER

        node_curv = [float(curv_by_eps[e][0][(vtx,)]) for e in eps_range]  # strict: must exist
        node_curv = np.array(node_curv, dtype=np.float32)
        node_curv = apply_sigmoid(node_curv, temp=10.0, center=True)
        vec = np.concatenate([atom_feat, node_curv])

        feat_dict["node"][vtx] = (vec, atom_context)
        Gnode.add_node(vtx, feat_key=vtx)  # pointer only

    # ---------------- Covalent edges with edge pointers ------------
    for b in mol_closed.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        vu, vv = mol2vtx[u], mol2vtx[v]
        i, j = (vu, vv) if vu < vv else (vv, vu)

        bond_feat = bond_fp(b, BOND_CFG)
        bond_context = bond_to_vocab(mol_closed, b) # added bond context from GROVER

        edge_curv = [float(curv_by_eps[e][1].get((i, j), -1.0e+4)) for e in eps_range]
        edge_curv = np.array(edge_curv, dtype=np.float32)
        edge_curv = apply_sigmoid(edge_curv, temp=10.0, center=True)
        vec = np.concatenate([bond_feat, edge_curv])

        feat_dict["edge"][(i, j)] = (vec, bond_context)
        Gnode.add_edge(i, j, feat_key=(i, j))  # pointer only

    # ---------------- Edge–edge graph (edges become nodes) ---------
    Gedge = nx.Graph()
    edges = list(Gnode.edges())
    edge_id = {e: idx for idx, e in enumerate(edges)}
    for idx, (i, j) in enumerate(edges):
        Gedge.add_node(idx, feat_key=(i, j))

    return Gnode, Gedge, feat_dict


def build_rips_graph(cutoff, mol_closed, vtx2mol, curv_by_eps, eps_range, min_dist):
    feat_dict = {
        "node": {},      # key: v (int Rips vertex)            -> np.ndarray
        "edge": {},      # key: (i,j) sorted                    -> np.ndarray
        "tri": {}        # key: (i,j,k) sorted                  -> np.ndarray
    }

    # ---------------- Node–node graph with node pointers ----------------
    Gnode = nx.Graph()
    for vtx, mol_idx in vtx2mol.items():
        atom_feat = atom_fp(mol_closed.GetAtomWithIdx(mol_idx), ATOM_CFG)
        node_curv = [float(curv_by_eps[e][0][(vtx,)]) for e in eps_range]
        node_curv = np.array(node_curv, dtype=np.float32)
        node_curv = apply_sigmoid(node_curv, temp=10.0, center=True)
        vec = np.concatenate([atom_feat, node_curv])
        feat_dict["node"][vtx] = vec
        Gnode.add_node(vtx, feat_key=vtx)  # pointer only

    # Edges/triangles (built globally; filter to the closed-molecule vertex set)
    edges_list, tris_list = edges_tris_for_cutoff(float(cutoff), curv_by_eps, min_dist)

    # ---------------- Add Rips edges with edge pointers -----------------
    for i, j in edges_list:
        assert i < j
        if i not in vtx2mol or j not in vtx2mol:
            continue

        u, v = vtx2mol[i], vtx2mol[j]
        bond = mol_closed.GetBondBetweenAtoms(u, v)
        if bond is not None:
            bond_feat = bond_fp(bond, BOND_CFG)
        else:
            bond_feat = np.zeros(BOND_CFG.n_features, dtype=np.float32)

        edge_curv_cols = [float(curv_by_eps[e][1].get((i, j), -1.0e+4)) for e in eps_range]
        edge_curv_cols = np.array(edge_curv_cols, dtype=np.float32)
        edge_curv_cols = apply_sigmoid(edge_curv_cols, temp=10.0, center=True)        
        vec = np.concatenate([bond_feat, edge_curv_cols])
        feat_dict["edge"][(i, j)] = vec
        Gnode.add_edge(i, j, feat_key=(i, j))  # pointer only

    # ---------------- Edge–edge graph (edges become nodes) --------------
    Gedge = nx.Graph()
    edges = list(Gnode.edges())
    edge_id = {e: idx for idx, e in enumerate(edges)}
    for idx, (u, v) in enumerate(edges):
        Gedge.add_node(idx, feat_key=(u,v))

    # ---------------- Triangles: single copy in feat_dict ----------------
    for i, j, k in tris_list:
        assert i < j < k
        if (i, j) not in edge_id or (j, k) not in edge_id or (i, k) not in edge_id:
            continue

        vals = [float(curv_by_eps[e][2].get((i, j, k), -1.0e+4)) for e in eps_range]
        tri_vec = np.array(vals, dtype=np.float32)
        tri_vec = apply_sigmoid(tri_vec, temp=10.0, center=True)                
        feat_dict["tri"][(i, j, k)] = tri_vec

        ix = edge_id[(i, j)]
        iy = edge_id[(j, k)]
        iz = edge_id[(i, k)]
        assert ix != iy and iy != iz and ix != iz

        # Attach triangle pointer (no vector duplication on each EE edge)
        Gedge.add_edge(ix, iy, feat_key=(i, j, k))
        Gedge.add_edge(iy, iz, feat_key=(i, j, k))
        Gedge.add_edge(ix, iz, feat_key=(i, j, k))

    return Gnode, Gedge, feat_dict
