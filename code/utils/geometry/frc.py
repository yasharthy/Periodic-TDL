import numpy as np
import gudhi
from itertools import combinations
import networkx as nx

from itertools import combinations

def frc_unweighted(points, edges, triangles):
    """
    Optimized unweighted Forman-Ricci curvature for nodes, edges, and triangles.
    points    : iterable of node ids  (0..N-1)
    edges     : list of sorted 2-tuples
    triangles : list of sorted 3-tuples
    Returns   : node_curv, edge_curv, tri_curv (dicts)
    """

    # --- 1. Build adjacency (no NetworkX) ---
    neighbors = {n: set() for n in points}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # --- 2. Convert triangles to frozensets for O(1) intersection ---
    triangles = [frozenset(tri) for tri in triangles]

    # --- 3. Build edge → triangles index ---
    tri_dict = {e: set() for e in edges}
    for tri in triangles:
        for e in combinations(sorted(tri), 2):  # tri is a frozenset
            tri_dict[e].add(tri)

    # --- 4. Precompute triangle → edges map ---
    tri_edges_map = {tri: list(combinations(sorted(tri), 2)) for tri in triangles}

    # --- 5. Optimized parallel-edge count ---
    def count_parallel_edges(e):
        u, v = e
        # all edges incident to u or v
        incident_count = (len(neighbors[u]) - 1) + (len(neighbors[v]) - 1)

        # subtract edges in triangles containing (u,v)
        tri_edges_seen = set()
        for tri in tri_dict[e]:
            tri_edges_seen.update(tri_edges_map[tri])
        tri_edges_seen.discard(e)

        return incident_count - len(tri_edges_seen)

    # --- 6. Edge curvature ---
    edge_curv = {}
    for e in edges:
        num_faces    = 2
        num_cofaces  = len(tri_dict[e])
        num_parallel = count_parallel_edges(e)
        edge_curv[e] = num_faces + num_cofaces - num_parallel

    # --- 7. Node curvature (avg over incident edges) ---
    node_curv = {}
    for n in points:
        inc_edges = [(n, nbr) if n < nbr else (nbr, n) for nbr in neighbors[n]]
        node_curv[(n,)] = (
            sum(edge_curv[e] for e in inc_edges) / len(inc_edges)
            if inc_edges else 0.0
        )

    # --- 8. Triangle curvature ---
    tri_curv = {}
    for tri in triangles:
        num_faces   = 3
        num_cofaces = 0

        # candidate triangles that share at least one edge
        cand_tris = set()
        for e in tri_edges_map[tri]:
            cand_tris.update(tri_dict[e])
        cand_tris.discard(tri)

        parallels = sum(1 for other in cand_tris if len(tri & other) == 2)
        tri_curv[tuple(sorted(tri))] = num_faces + num_cofaces - parallels

    return node_curv, edge_curv, tri_curv


def rips_frc(points, epsilon):
    rips = gudhi.RipsComplex(points=points, max_edge_length=epsilon)
    st   = rips.create_simplex_tree(max_dimension=3)

    simplices = [tuple(s) for s, _ in st.get_skeleton(3)]
    nodes     = [s[0]             for s in simplices if len(s) == 1]
    edges     = [tuple(sorted(s)) for s in simplices if len(s) == 2]
    triangles = [tuple(sorted(s)) for s in simplices if len(s) == 3]

    # return nodes, edges, triangles
    return frc_unweighted(nodes, edges, triangles)


# ---- Gudhi-based FRC from a PRECOMPUTED distance matrix (dicts only) ----
def frc_from_min_dist_gudhi_dict(min_dist, epsilon: float):
    rips = gudhi.RipsComplex(distance_matrix=min_dist, max_edge_length=epsilon)
    st   = rips.create_simplex_tree(max_dimension=3)

    simplices = [tuple(s) for s, _ in st.get_skeleton(3)]
    nodes     = [s[0]             for s in simplices if len(s) == 1]
    edges     = [tuple(sorted(s)) for s in simplices if len(s) == 2]
    triangles = [tuple(sorted(s)) for s in simplices if len(s) == 3]

    # print(f"Nodes: {len(nodes)};\t Edges: {len(edges)};\t Triangles: {len(triangles)}")
    # print(len(edges))
    # print(f"Computing curvature at {epsilon}")
    node_curv, edge_curv, tri_curv = frc_unweighted(nodes, edges, triangles)
    return node_curv, edge_curv, tri_curv


# === Test script ===
if __name__ == "__main__":
    np.random.seed(42)
    points = np.random.rand(6, 3) * 5.0  # 6 points in 3D, scaled up to spread out

    epsilon = 4.0

    node_curv, edge_curv, tri_curv = rips_frc(points, epsilon)

    print("3D Coordinates:\n", points)
    print("\n--- Node Curvatures ---")
    for k, v in node_curv.items():
        print(f"{k}: {v:.3f}")

    print("\n--- Edge Curvatures ---")
    for k, v in edge_curv.items():
        print(f"{k}: {v:.3f}")

    print("\n--- Triangle Curvatures ---")
    for k, v in tri_curv.items():
        print(f"{k}: {v:.3f}")

# import time 

# # Create 100 random 3D points
# np.random.seed(42)
# points_3d = np.random.rand(100, 3) * 5.0

# # Set a large epsilon so the Rips complex is dense
# epsilon = 3.0

# # Time repeated curvature computations
# start_time = time.time()
# for _ in range(100):
#     rips_frc(points_3d, epsilon)  # or frc_unweighted_nx
# elapsed = time.time() - start_time

# print(f"Total time for 100 repeats: {elapsed:.2f} seconds")