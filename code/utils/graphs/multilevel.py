import numpy as np

from utils.geometry.periodic_repn import compute_pairwise_distances, compute_min_distance_across_augmentations

from utils.graphs.helpers import compute_index_maps, compute_curvatures, cross_maps
from utils.graphs.builders import build_closed_graph, build_rips_graph


class EpsConfig:
    def __init__(self, k1=None, k2=None, k3=None):
        self.k1 = list(k1) if k1 is not None else list(np.arange(2.0, 3.0 + 1e-12, 0.25))
        self.k2 = list(k2) if k2 is not None else list(np.arange(3.0, 4.0 + 1e-12, 0.25))
        self.k3 = list(k3) if k3 is not None else list(np.arange(4.0, 5.0 + 1e-12, 0.25))

    @property
    def all(self):
        return sorted(set(self.k1 + self.k2 + self.k3))


def build_multilevel_complex(mol_closed, coords, ref_atoms, oid_list, eps_cfg: EpsConfig):
    mol2oid, oid2v, vtx2mol, mol2vtx = compute_index_maps(mol_closed, ref_atoms, oid_list)
    min_dist = compute_min_distance_across_augmentations(compute_pairwise_distances(coords))
    curv_by_eps = compute_curvatures(min_dist, eps_cfg.all)

    G1n, G1e, F1 = build_closed_graph(mol_closed, mol2vtx, curv_by_eps, eps_cfg.k1)
    G2n, G2e, F2 = build_rips_graph(3.0, mol_closed, vtx2mol, curv_by_eps, eps_cfg.k2, min_dist)
    G3n, G3e, F3 = build_rips_graph(4.0, mol_closed, vtx2mol, curv_by_eps, eps_cfg.k3, min_dist)

    # --- replace the return at end of build_multilevel_complex(...) ---
    levels={"k1":{"node_graph":G1n,"edge_graph":G1e,"feat_dict":F1},
            "k2":{"node_graph":G2n,"edge_graph":G2e,"feat_dict":F2},
            "k3":{"node_graph":G3n,"edge_graph":G3e,"feat_dict":F3}}
    
    cross={"k3->k2":cross_maps(levels["k3"],levels["k2"]),
        "k2->k1":cross_maps(levels["k2"],levels["k1"])}
    
    return {"levels":levels, "cross":cross}