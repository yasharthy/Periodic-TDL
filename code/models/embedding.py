"""
Multi-head GENStacks from embedding_v3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_mean_pool
from torch_geometric.nn import GraphNorm

# ---------- Utilities ----------
class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU, drop=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [act(), nn.Dropout(drop)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------- GENConv Stack (used for node↔node or edge↔edge) ----------
class GENStack(nn.Module):
    def __init__(self, hidden: int, attr_in: int, depth: int = 2, drop: float = 0.0):
        super().__init__()
        self.depth = depth
        self.attr_proj = nn.ModuleList([nn.Linear(attr_in, hidden) for _ in range(depth)])
        # Use GENConv with softmax aggregation and instance normalization to improve expressivity.
        # Set learnable temperature on the softmax aggregation (learn_t=True) and enable message
        # normalization with learnable scaling (msg_norm=True, learn_msg_scale=True). Using
        # InstanceNorm ('instance') for the internal MLP layers inside GENConv helps stabilize
        # training and often leads to better generalization. An additive bias is enabled in the
        # MLP layers to increase representational capacity.
        self.convs = nn.ModuleList([
            GENConv(hidden, hidden, aggr='softmax',  # use softmax aggregation rather than mean
                        t=1.0, learn_t=True, msg_norm=True, learn_msg_scale=True,
                        norm='instance', num_layers=2, expansion=2, eps=1e-7, bias=True)
                    for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.drop = nn.Dropout(drop)
    def forward(self, x, edge_index, edge_attr):
        # if self.depth == 0 or edge_index.numel() == 0:
        #     return x
        h = x
        for l in range(self.depth):
            # project edge attributes
            ea = self.attr_proj[l](edge_attr) if edge_attr.numel() else edge_attr

            # pre-activation normalization + activation
            pre = self.norms[l](h)         # GraphNorm or LayerNorm
            pre = F.relu(pre)
            pre = self.drop(pre)

            # message passing
            out = self.convs[l](pre, edge_index, ea)

            # residual connection
            h = h + out
        return h
    
class MultiHeadGENStack(nn.Module):
    def __init__(self, hidden: int, attr_in: int, depth: int, drop: float, num_heads: int = 4):
        super().__init__()
        assert hidden % num_heads == 0
        self.head_dim = hidden // num_heads
        self.num_heads = num_heads
        # Project full hidden vector into head-specific subspaces
        self.in_proj = nn.Linear(hidden, hidden)  # produces num_heads * head_dim
        self.heads = nn.ModuleList([
            GENStack(self.head_dim, attr_in=attr_in, depth=depth, drop=drop)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x, edge_index, edge_attr):
        # apply learnable projection and reshape for heads
        projected = self.in_proj(x)  # [N, hidden]
        xs = projected.view(x.size(0), self.num_heads, self.head_dim)
        outs = []
        for i, head in enumerate(self.heads):
            outs.append(head(xs[:, i], edge_index, edge_attr))
        h_cat = torch.cat(outs, dim=-1)
        return self.out_proj(h_cat)

# ---------- Edge→Node Fusion ----------
class EdgeToNodeFusion(nn.Module):
    def __init__(self, edge_attr_in: int, edge_x_in: int, fused_out: int):
        super().__init__()
        self.mlp = nn.Linear(edge_attr_in + edge_x_in, fused_out)
    def forward(self, edge_attr, edge_node_idx, x_edge):
        if edge_attr.numel() == 0:
            return edge_attr
        z = x_edge[edge_node_idx]  # [E, H]
        ea = torch.cat([edge_attr, z], dim=-1)
        return self.mlp(ea)

# ---------- Down‑map (top→down gated residual) ----------
class DownMap(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.msg = MLP([hidden, hidden])
        self.gate = MLP([2*hidden, hidden], act=nn.Sigmoid)
    def forward(self, src_x, dst_x, edge_index):
        if edge_index.numel() == 0:
            return dst_x
        si, di = edge_index
        m = self.msg(src_x[si])
        g = self.gate(torch.cat([dst_x[di], m], dim=-1))
        out = dst_x.clone()
        out.index_add_(0, di, g * m)
        return out

# ---------- Orchestrator ----------
class HSMPEmbedding(nn.Module):
    def __init__(self, sample, hidden=128, drop=0.0,
                 L_edge_k3=2, L_node_k3=2,
                 L_edge_k2=2, L_node_k2=2,
                 L_node_k1=2, num_heads=12):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        
        # 1) Projections
        self.node_proj = nn.ModuleDict({k: MLP([sample[k].x.size(-1), hidden, hidden], drop=drop)
            for k in ['node_k1', 'node_k2', 'node_k3']})
        self.edge_proj = nn.ModuleDict({k: MLP([sample[k].x.size(-1), hidden, hidden], drop=drop)
            for k in ['edge_k1', 'edge_k2', 'edge_k3']})
        # define per-level GraphNorm modules
        self.node_norm = nn.ModuleDict({k: GraphNorm(hidden) for k in ['node_k1', 'node_k2', 'node_k3']})
        self.edge_norm = nn.ModuleDict({k: GraphNorm(hidden) for k in ['edge_k1', 'edge_k2', 'edge_k3']})

        # 2) Stacks per level
        # Edge stacks use triangle attrs sizes; Node stacks use fused attr sizes (hidden after fusion)
        def tri_attr_dim(k):
            rel = (f'edge_k{k}', 'shared_tri', f'edge_k{k}')
            return sample[rel].edge_attr.size(-1) if 'edge_attr' in sample[rel] and sample[rel].edge_attr.numel() else 0
        def edge_attr_dim(k):
            rel = (f'node_k{k}', 'via_edge', f'node_k{k}')
            return sample[rel].edge_attr.size(-1)

        self.edge_k3 = MultiHeadGENStack(hidden, attr_in=max(1, tri_attr_dim(3)), depth=L_edge_k3, drop=drop, num_heads=self.num_heads)
        self.edge_k2 = MultiHeadGENStack(hidden, attr_in=max(1, tri_attr_dim(2)), depth=L_edge_k2, drop=drop, num_heads=self.num_heads)

        self.node_k3 = MultiHeadGENStack(hidden, attr_in=hidden, depth=L_node_k3, drop=drop, num_heads=self.num_heads)
        self.node_k2 = MultiHeadGENStack(hidden, attr_in=hidden, depth=L_node_k2, drop=drop, num_heads=self.num_heads)
        self.node_k1 = MultiHeadGENStack(hidden, attr_in=hidden, depth=L_node_k1, drop=drop, num_heads=self.num_heads)

        # 3) Fusion modules (edge_attr + edge_x → fused attr of size H)
        self.fuse_k3 = EdgeToNodeFusion(edge_attr_in=edge_attr_dim(3), edge_x_in=hidden, fused_out=hidden)
        self.fuse_k2 = EdgeToNodeFusion(edge_attr_in=edge_attr_dim(2), edge_x_in=hidden, fused_out=hidden)
        self.fuse_k1 = EdgeToNodeFusion(edge_attr_in=edge_attr_dim(1), edge_x_in=hidden, fused_out=hidden)

        # 4) Down‑maps
        self.down_node_32 = DownMap(hidden); self.down_edge_32 = DownMap(hidden)
        self.down_node_21 = DownMap(hidden); self.down_edge_21 = DownMap(hidden)

    def forward(self, data):
        # apply projection, then GraphNorm with batch info
        x_node = {}
        x_edge = {}
        for k in (1, 2, 3):
            ntype = f'node_k{k}'
            etype = f'edge_k{k}'
            x_node[k] = self.node_proj[ntype](data[ntype].x)
            x_node[k] = self.node_norm[ntype](x_node[k], data[ntype].batch)  # pass batch
            x_edge[k] = self.edge_proj[etype](data[etype].x)
            x_edge[k] = self.edge_norm[etype](x_edge[k], data[etype].batch)  # pass batch

        # --- helper: align local edge_node_idx -> global rows in x_edge[k] and assert batch-safety ---
        def _align_idx(k: int):
            ntype = f'node_k{k}'
            etype = f'edge_k{k}'
            rel   = (ntype, 'via_edge', ntype)

            idx = data[rel].edge_node_idx
            if idx.numel() == 0:
                return idx  # nothing to align

            ei = data[rel].edge_index
            nb = data[ntype].batch          # sample id per node
            eb = data[etype].batch          # sample id per edge-node

            counts = torch.bincount(eb, minlength=int(nb.max().item()) + 1)
            starts = torch.cat([counts.new_zeros(1, device=counts.device), counts.cumsum(0)[:-1]])
            idx_g  = idx + starts[nb[ei[0]]]

            # Asserts (simple & strict)
            assert idx_g.min().item() >= 0 and idx_g.max().item() < x_edge[k].size(0), f"[k={k}] edge_node_idx out of bounds"
            assert torch.all(nb[ei[0]] == eb[idx_g]).item(), f"[k={k}] fusion crosses batch slices"
            return idx_g

        # ----- k3 -----
        rel_e3 = ("edge_k3", "shared_tri", "edge_k3")
        rel_n3 = ("node_k3", "via_edge", "node_k3")

        ea_nn3 = data[rel_n3].edge_attr
        idx3   = data[rel_n3].edge_node_idx
        ei3    = data[rel_n3].edge_index

        # (debug: keep for later – uncomment when probing)
        # print(f"[k3] ea_nn3={tuple(ea_nn3.shape)}  idx3={idx3.numel()}  x_edge[3] rows={x_edge[3].size(0)}")
        # nb3 = data["node_k3"].batch; eb3 = data["edge_k3"].batch
        # print("WTH is a batch?", nb3.shape, eb3.shape)
        # print(ei3[0])
        # mism_before = (nb3[ei3[0]] != eb3[idx3]).sum().item()
        # print(f"[k3] cross-batch mismatches BEFORE: {mism_before} / {idx3.numel()}")

        idx3 = _align_idx(3)

        # Edge-edge MP on k3
        if data[rel_e3].edge_index.numel():
            ea3 = data[rel_e3].edge_attr if 'edge_attr' in data[rel_e3] else torch.zeros((data[rel_e3].edge_index.size(1), 1), device=x_edge[3].device)
            x_edge[3] = self.edge_k3(x_edge[3], data[rel_e3].edge_index, ea3)

        # Fuse → node GEN (k3)
        ea_f3     = self.fuse_k3(ea_nn3, idx3, x_edge[3])
        x_node[3] = self.node_k3(x_node[3], data[rel_n3].edge_index, ea_f3)

        # Down to k2
        rel_dn_n32 = ("node_k3", "down", "node_k2")
        rel_dn_e32 = ("edge_k3", "down", "edge_k2")
        x_node[2]  = self.down_node_32(x_node[3], x_node[2], data[rel_dn_n32].edge_index)
        x_edge[2]  = self.down_edge_32(x_edge[3], x_edge[2], data[rel_dn_e32].edge_index)

        # ----- k2 -----
        rel_e2 = ("edge_k2", "shared_tri", "edge_k2")
        if data[rel_e2].edge_index.numel():
            ea2 = data[rel_e2].edge_attr if 'edge_attr' in data[rel_e2] else torch.zeros((data[rel_e2].edge_index.size(1), 1), device=x_edge[2].device)
            x_edge[2] = self.edge_k2(x_edge[2], data[rel_e2].edge_index, ea2)

        rel_n2 = ("node_k2", "via_edge", "node_k2")
        ea_nn2 = data[rel_n2].edge_attr
        idx2   = _align_idx(2)  # assert inside ensures batch-safe fusion (k2)

        ea_f2     = self.fuse_k2(ea_nn2, idx2, x_edge[2])
        x_node[2] = self.node_k2(x_node[2], data[rel_n2].edge_index, ea_f2)

        # Down to k1
        rel_dn_n21 = ("node_k2", "down", "node_k1")
        rel_dn_e21 = ("edge_k2", "down", "edge_k1")
        x_node[1]  = self.down_node_21(x_node[2], x_node[1], data[rel_dn_n21].edge_index)
        x_edge[1]  = self.down_edge_21(x_edge[2], x_edge[1], data[rel_dn_e21].edge_index)

        # ----- k1 ----- (no shared_tri MP)
        rel_n1 = ("node_k1", "via_edge", "node_k1")
        ea_nn1 = data[rel_n1].edge_attr
        idx1   = _align_idx(1)  # assert inside ensures batch-safe fusion (k1)

        ea_f1     = self.fuse_k1(ea_nn1, idx1, x_edge[1])
        x_node[1] = self.node_k1(x_node[1], data[rel_n1].edge_index, ea_f1)

        # Readout
        batch = getattr(data['node_k1'], 'batch', None)
        if batch is None:
            raise RuntimeError("Missing batch assignment for node_k1; cannot pool per-graph.")

        pooled = global_mean_pool(x_node[1], batch)

        return {"atom_repr": x_node[1], "bond_repr": ea_f1, "graph_repr": pooled, "batch": batch}