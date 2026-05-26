import torch
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def load_dataset(path):
    obj = torch.load(path, map_location="cpu")
    assert isinstance(obj, list), "Expected list[HeteroData] (not a dict)."
    assert len(obj) > 0, "Empty dataset."
    return obj

def _load_one(ci, path):
    data = load_dataset(path)  # returns list[HeteroData]
    return ci, data

def load_all_parallel_proc(HETERO_DIR, datafile, n_parts, max_workers):
    paths = []
    for ci in range(n_parts):
        p = os.path.join(HETERO_DIR, f"{datafile}_hetero_part{ci:02d}.pt")
        if os.path.exists(p):
            paths.append((ci, p))

    graphs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_load_one, ci, p) for ci, p in paths]
        for fut in as_completed(futs):
            ci, part_graphs = fut.result()
            print(f"Loaded part {ci:02d}: {len(part_graphs)} heterographs")
            graphs.extend(part_graphs)

    print(f"Total heterographs loaded: {len(graphs)}")
    return graphs


def sanitize_feature_dims(dataset):
    """For some Rips complexes that do not have triangles"""
    # discover max feature dims per relation across dataset
    rel_F = defaultdict(int)
    for g in dataset:
        for rel in g.edge_types:
            if 'edge_attr' in g[rel]:
                F = int(g[rel].edge_attr.size(-1))
                rel_F[rel] = max(rel_F[rel], F)

    # pad/standardize each graph
    for g in dataset:
        for rel in g.edge_types:
            if 'edge_attr' in g[rel]:
                ea = g[rel].edge_attr
                Fexp = max(1, rel_F[rel])
                if ea.size(-1) == 0:
                    g[rel].edge_attr = torch.zeros((ea.size(0), Fexp), dtype=torch.float32)
                elif ea.size(-1) < Fexp:
                    pad = torch.zeros((ea.size(0), Fexp - ea.size(1)), dtype=ea.dtype, device=ea.device)
                    g[rel].edge_attr = torch.cat([ea, pad], dim=1)
                elif ea.size(-1) > Fexp:
                    g[rel].edge_attr = ea[:, :Fexp]