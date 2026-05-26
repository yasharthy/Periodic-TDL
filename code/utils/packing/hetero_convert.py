"""
Compact PyG HeteroData conversion for the multilevel complex.
Assumptions (per spec):
1) One global node order shared by all levels.
2) Edge-graph node ids are contiguous per level; use as-is.
3) Triangles are not separate stores; their features are edge_attr on edge↔edge.
4) Only node/edge cross-level maps are used (no tri cross).

Input shape (from build_multilevel_complex):
{
  "levels": {
    "k1": {"node_graph": NG1, "edge_graph": EG1, "feat_dict": F1},
    "k2": {"node_graph": NG2, "edge_graph": EG2, "feat_dict": F2},
    "k3": {"node_graph": NG3, "edge_graph": EG3, "feat_dict": F3},
  },
  "cross": {
    "k3->k2": {"edge": (2, M32) or list[(src_eid, dst_eid), ...]},
    "k2->k1": {"edge": (2, M21) or list[...] }
  }
}
"""
# utils/packing/hetero_convert.py
# utils/packing/hetero_convert.py (top of file, after imports)

from typing import Any, Iterable, List, Sequence, Tuple, Dict
import numpy as np
import torch
from torch_geometric.data import HeteroData

def _split_vec_ctx(entry: Any, need_ctx: bool) -> Tuple[np.ndarray, str | None]:
    """
    Accepts either:
      - vec : np.ndarray-like
      - (vec, ctx) : tuple
    Returns:
      - vec_np : np.ndarray (float32)
      - ctx    : str | None   (only if need_ctx=True and entry carries ctx)
    """
    if isinstance(entry, tuple):
        vec, ctx = entry
        vec_np = np.asarray(vec, dtype=np.float32)
        return vec_np, (ctx if need_ctx else None)
    else:
        vec_np = np.asarray(entry, dtype=np.float32)
        return vec_np, None


def _gather_by_keys(
    F: Dict[Any, Any],
    keys: Sequence[Any],
    want_ctx: bool
) -> Tuple[np.ndarray, List[str] | None]:
    """
    For a feature dict F (node or edge), gather entries in `keys` order.
    Each F[key] can be vec or (vec, ctx). Returns:
      - X : np.ndarray [len(keys), F_dim]
      - ctx_list : list[str] or None (only if want_ctx=True and at least one ctx is present)
    """
    X_list, ctx_list = [], []
    has_ctx = False
    for k in keys:
        vec_np, ctx = _split_vec_ctx(F[k], need_ctx=want_ctx)
        X_list.append(vec_np)
        if want_ctx:
            # Collect string if present; else append empty string to keep alignment
            if ctx is not None:
                ctx_list.append(ctx)
                has_ctx = True
            else:
                ctx_list.append("")  # placeholder so lengths stay consistent
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, 0), dtype=np.float32)
    return X, (ctx_list if (want_ctx and has_ctx) else None)


def _duplicate_undirected(
    ei: torch.Tensor,           # (2, E)
    ea: torch.Tensor,           # (E, F)
    edge_ctx_unique: List[str] | None
) -> Tuple[torch.Tensor, torch.Tensor, List[str] | None]:
    """
    Make edges undirected by concatenating the reversed edges and attrs.
    Duplicate the ctx list in the same order if provided.
    """
    ei_ud = torch.cat([ei, ei.flip(0)], dim=1)
    ea_ud = torch.cat([ea, ea], dim=0)
    if edge_ctx_unique is not None:
        ctx_ud = edge_ctx_unique + edge_ctx_unique
    else:
        ctx_ud = None
    return ei_ud, ea_ud, ctx_ud


def to_heterodata_from_complex(
    cpx: Dict[str, dict],
    include_atom_ctx: bool = False,
    include_bond_ctx: bool = False,
) -> HeteroData:
    levels, cross = cpx["levels"], cpx["cross"]
    Ks = sorted(levels, key=lambda s: int(s[1:]))  # ['k1','k2','k3']

    # Global node order shared by all levels
    base_nodes = sorted(levels[Ks[0]]["node_graph"].nodes())
    node2idx = {int(v): i for i, v in enumerate(base_nodes)}
    N = len(base_nodes)

    data = HeteroData()

    for k in Ks:
        L = levels[k]
        ntype = f"node_{k}"
        etype = f"edge_{k}"
        Fn, Fe, Ftri = L["feat_dict"]["node"], L["feat_dict"]["edge"], L["feat_dict"]["tri"]

        # ---------- Nodes ----------
        want_atom_ctx = (k == 'k1') and include_atom_ctx
        # Gather node features in the global base_nodes order
        Xn, node_ctx = _gather_by_keys(Fn, [int(v) for v in base_nodes], want_ctx=want_atom_ctx)
        data[ntype].x = torch.from_numpy(Xn)
        data[ntype].num_nodes = N
        if want_atom_ctx:
            data[ntype].ctx_label = node_ctx

        # ---------- Edge-as-node (line graph nodes) ----------
        EG = L["edge_graph"]
        e_ids = sorted(EG.nodes())
        edge_keys = [tuple(EG.nodes[e]["feat_key"]) for e in e_ids]  # canonical (i<j)

        # For edge-as-node features, use only vec part even if (vec, ctx) is present at k1
        Xe, _ = _gather_by_keys(Fe, edge_keys, want_ctx=False)
        data[etype].x = torch.from_numpy(Xe)
        data[etype].num_nodes = int(Xe.shape[0])

        # ---------- Node↔node relation (edge_attr + optional ctx) ----------
        # Map closed-graph vertex ids (edge_keys pairs) → global node indices
        nn_pairs = [(node2idx[int(i)], node2idx[int(j)]) for (i, j) in edge_keys]
        nn_ei = torch.from_numpy(np.array(nn_pairs, dtype=np.int64).T)

        # Collect edge_attr vecs and, if requested at k1, the ctx strings in the unique-edge order
        want_bond_ctx = (k == 'k1') and include_bond_ctx
        Ea_unique, edge_ctx_unique = _gather_by_keys(Fe, [(int(i), int(j)) for (i, j) in edge_keys],
                                                     want_ctx=want_bond_ctx)
        ea = torch.from_numpy(Ea_unique)

        # Make undirected and mirror ctx accordingly
        ei, ea, edge_ctx = _duplicate_undirected(nn_ei, ea, edge_ctx_unique)

        rel = (ntype, 'via_edge', ntype)
        data[rel].edge_index = ei
        data[rel].edge_attr  = ea

        # Keep your existing fusion alignment index (NN edges → edge-as-node rows)
        idx = torch.arange(len(edge_keys), dtype=torch.long)
        data[rel].edge_node_idx = torch.cat([idx, idx], dim=0)

        if want_bond_ctx:
            data[rel].ctx_label = edge_ctx

        # ---------- Edge↔edge relation (triangle attrs) ----------
        if EG.number_of_edges():
            ee_pairs, ee_attr = [], []
            for u, v, d in EG.edges(data=True):
                vec = Ftri[tuple(d['feat_key'])]  # triangles only store vecs
                ee_pairs.append((int(u), int(v)))
                ee_attr.append(vec)
            ee = torch.from_numpy(np.array(ee_pairs, np.int64).T)
            ea_tri = torch.from_numpy(np.asarray(ee_attr, np.float32))

            # undirected duplication
            ee = torch.cat([ee, ee.flip(0)], dim=1)
            ea_tri = torch.cat([ea_tri, ea_tri], dim=0)

            data[(etype, 'shared_tri', etype)].edge_index = ee
            data[(etype, 'shared_tri', etype)].edge_attr  = ea_tri
        else:
            data[(etype, 'shared_tri', etype)].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[(etype, 'shared_tri', etype)].edge_attr  = torch.empty((0, 0), dtype=torch.float32)

    # ---------- Cross-level down maps ----------
    for src, dst in zip(Ks[::-1][:-1], Ks[::-1][1:]):
        key = f"{src}->{dst}"
        nmap = np.asarray(cross[key]['node'], dtype=np.int64)
        src_ids = torch.tensor([node2idx[int(v)] for v in nmap[0]], dtype=torch.long)
        dst_ids = torch.tensor([node2idx[int(v)] for v in nmap[1]], dtype=torch.long)
        data[(f'node_{src}', 'down', f'node_{dst}')].edge_index = torch.stack([src_ids, dst_ids])

        arr = np.asarray(cross[key]['edge'], dtype=np.int64)
        data[(f'edge_{src}', 'down', f'edge_{dst}')].edge_index = torch.from_numpy(arr)

    return data







# from typing import Dict
# import numpy as np
# import torch
# from torch_geometric.data import HeteroData


# def to_heterodata_from_complex(cpx: Dict[str, dict]) -> HeteroData:
#     levels, cross = cpx["levels"], cpx.get("cross", {})
#     Ks = sorted(levels, key=lambda s: int(s[1:]))              # ['k1','k2','k3']

#     # Global node order shared by all levels
#     base_nodes = sorted(levels[Ks[0]]["node_graph"].nodes())
#     node2idx = {int(v): i for i, v in enumerate(base_nodes)}
#     N = len(base_nodes)

#     data = HeteroData()

#     for k in Ks:
#         L = levels[k]
#         ntype = f"node_{k}"
#         etype = f"edge_{k}"
#         Fn, Fe, Ftri = L["feat_dict"]["node"], L["feat_dict"]["edge"], L["feat_dict"]["tri"]

#         # Node features: global order
#         xn_np = np.asarray([Fn[int(v)] for v in base_nodes], dtype=np.float32)
#         data[ntype].x = torch.from_numpy(xn_np)
#         data[ntype].num_nodes = N

#         # Edge-as-node features: order different for each k_i
#         EG = L["edge_graph"]
#         e_ids = sorted(EG.nodes())
#         edge_keys = [tuple(EG.nodes[e]["feat_key"]) for e in e_ids]  # ensure tuple, canonical (i<j)
#         xe = np.asarray([Fe[e_key] for e_key in edge_keys], dtype=np.float32)
#         xe = torch.from_numpy(xe)
#         data[etype].x, data[etype].num_nodes = xe, int(xe.shape[0])

#         # Node↔node within-level (edge_attr = edge features of (i,j))
#         NG = L["node_graph"]
#         nn_ei = np.array([(node2idx[int(i)], node2idx[int(j)]) for i, j in edge_keys], dtype=np.int64).T
#         nn_ea = np.asarray([Fe[(int(i), int(j))] for i, j in edge_keys], dtype=np.float32)

#         ei = torch.from_numpy(nn_ei)            # shape (2, E)
#         ea = torch.from_numpy(nn_ea)            # shape (E, F_edge)
#         idx = torch.arange(len(edge_keys), dtype = torch.long)      # maps NN edges → EG node rows

#         # --- make undirected by duplicating reverse edges ---
#         ei = torch.cat([ei, ei.flip(0)], dim=1)   # [u,v] and [v,u]
#         ea = torch.cat([ea, ea], dim=0)           # duplicate attributes
#         idx = torch.cat([idx, idx], dim=0)        # keep fusion alignment

#         data[(ntype, 'via_edge', ntype)].edge_index   = ei
#         data[(ntype, 'via_edge', ntype)].edge_attr    = ea
#         data[(ntype, 'via_edge', ntype)].edge_node_idx = idx

#         # Edge↔edge within-level (edge_attr = triangle features)
#         if EG.number_of_edges():
#             ee_pairs, ee_attr = [], []
#             for u, v, d in EG.edges(data=True):
#                 vec = Ftri[tuple(d['feat_key'])]   # your current lookup
#                 # vec = Ftri[tuple(d.get('feat_key'))]   # your current lookup
#                 ee_pairs.append((int(u), int(v)))
#                 ee_attr.append(vec)

#             ee = torch.from_numpy(np.array(ee_pairs, np.int64).T)   # (2, EE)
#             ea = torch.from_numpy(np.asarray(ee_attr, np.float32))  # (EE, F_tri)

#             # --- make undirected ---
#             ee = torch.cat([ee, ee.flip(0)], dim=1)
#             ea = torch.cat([ea, ea], dim=0)

#             data[(etype, 'shared_tri', etype)].edge_index = ee
#             data[(etype, 'shared_tri', etype)].edge_attr  = ea
#         else:
#             data[(etype, 'shared_tri', etype)].edge_index = torch.empty((2, 0), dtype=torch.long)
#             data[(etype, 'shared_tri', etype)].edge_attr  = torch.empty((0, 0), dtype=torch.float32)

#     # Cross-level (top→down): use provided cross maps for nodes & edges
#     for src, dst in zip(Ks[::-1][:-1], Ks[::-1][1:]):
#         key = f"{src}->{dst}"
#         # Node cross-map provided as original vertex ids → map via global node2idx
#         nmap = np.asarray(cross[key]['node'], dtype=np.int64)
#         src_ids = torch.tensor([node2idx[int(v)] for v in nmap[0]], dtype=torch.long)
#         dst_ids = torch.tensor([node2idx[int(v)] for v in nmap[1]], dtype=torch.long)
#         data[(f'node_{src}', 'down', f'node_{dst}')].edge_index = torch.stack([src_ids, dst_ids])
    
#         # Edge cross-map: local line-graph ids as provided
#         arr = np.asarray(cross[key]['edge'], dtype=np.int64)
#         ei = torch.from_numpy(arr)
#         data[(f'edge_{src}', 'down', f'edge_{dst}')].edge_index = ei
    
#     return data
