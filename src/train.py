import os
import json
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GATConv

from .preprocess import load_data

# ------------------ HASHPIPE core -------------------------------------------------
class HashPipeFilter(nn.Module):
    """Multi-resolution SimHash + Progressive Top-K Cache as described in the paper.
    The module is kept self-contained so it can be plugged into any GNN layer that
    needs to prune an adjacency list before computing attention scores.
    """

    def __init__(self, top_k: int = 32, cache_size: int = 16, sketch_dim: int = 32,
                 thresholds=(4, 8, 12)) -> None:
        super().__init__()
        self.T = top_k
        self.cache_size = cache_size
        self.s = sketch_dim
        self.t16, self.t32, self.t64 = thresholds
        # the cache is stored as a python list of torch tensors on CPU to avoid GPU RAM use
        self.register_buffer("_dummy", torch.empty(0))  # buffer to obtain correct device
        self._cache = []  # type: ignore

    # -------------------------------------------------------------------------
    @staticmethod
    def _view_as_uint(x: Tensor) -> Tensor:
        # ensure contiguous and convert the first N bytes to uint8 for hashing
        return x.view(torch.uint8)

    # -------------------------------------------------------------------------
    def forward(self, row: Tensor, col: Tensor, q: Tensor, k: Tensor) -> Tensor:  # noqa: C901
        """Return a boolean mask indicating which (row[i], col[i]) pairs survive.

        Args
        -----
        row:  (E,)  source node indices for every edge in the CSR representation
        col:  (E,)  destination node indices
        q:    (N, d)  Query vectors (usually from the current node representations)
        k:    (N, d)  Key   vectors (usually from neighbour node representations)
        """
        device = q.device
        if not self._cache:
            # lazily initialise caches
            self._cache = [torch.empty(0, dtype=torch.long, device='cpu') for _ in range(q.size(0))]

        mask = torch.zeros_like(col, dtype=torch.bool, device=device)

        # ---------------- Stage-1 : 16-bit SimHash --------------------------
        hq16 = self._view_as_uint(q[row][:, :16]).sum(-1).to(torch.int32)
        hk16 = self._view_as_uint(k[col][:, :16]).sum(-1).to(torch.int32)
        cand = (hq16 ^ hk16).bit_count() <= self.t16

        # ---------------- Stage-2 : 32-bit SimHash --------------------------
        if cand.any():
            idx = cand.nonzero(as_tuple=False).squeeze()
            hq32 = self._view_as_uint(q[row[idx]][:, :32]).sum(-1).to(torch.int32)
            hk32 = self._view_as_uint(k[col[idx]][:, :32]).sum(-1).to(torch.int32)
            cand[idx] &= (hq32 ^ hk32).bit_count() <= self.t32

        # ---------------- Stage-3 : 64-bit SimHash --------------------------
        idx = cand.nonzero(as_tuple=False).squeeze()
        if idx.numel() > 0:
            hq64 = self._view_as_uint(q[row[idx]]).sum(-1).to(torch.int64)
            hk64 = self._view_as_uint(k[col[idx]]).sum(-1).to(torch.int64)
            cand[idx] &= (hq64 ^ hk64).bit_count() <= self.t64

        mask |= cand.to(device)

        # ---------------- Union with Historical Cache -----------------------
        for i in range(row.numel()):
            src = row[i].item()
            if self._cache[src].numel():
                mask[i] |= (col[i] in self._cache[src])

        return mask

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update_cache(self, src_nodes: Tensor, dst_nodes: Tensor, scores: Tensor) -> None:
        """After an epoch, update PTKC (top-T cache) with the highest-scoring edges."""
        for v in src_nodes.unique():
            v = v.item()
            # pick top-T dst for this v
            mask = (src_nodes == v)
            if mask.any():
                topk = torch.topk(scores[mask], self.cache_size, largest=True, sorted=False).indices
                self._cache[v] = dst_nodes[mask][topk].cpu()


# =============================================================================
#  Simple GAT Model with optional HASHPIPE pruning
# =============================================================================
class GATWithHashPipe(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden: int, num_layers: int,
                 heads: int, use_hashpipe: bool = True, **hp_kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_hashpipe = use_hashpipe
        self.hp_filter = HashPipeFilter(**hp_kwargs) if use_hashpipe else None

        in_channels = num_features
        for i in range(num_layers):
            out_channels = hidden if i < num_layers - 1 else num_classes
            self.layers.append(GATConv(in_channels, out_channels // heads, heads=heads, dropout=0.6))
            in_channels = out_channels

    # ---------------------------------------------------------------------
    def forward(self, data):  # data is a torch_geometric.data.Data object
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            if self.use_hashpipe:
                edge_index = self._prune_edges(edge_index, x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        # last layer
        conv = self.layers[-1]
        if self.use_hashpipe:
            edge_index = self._prune_edges(edge_index, x)
        x = conv(x, edge_index)
        return F.log_softmax(x, dim=-1)

    # ---------------------------------------------------------------------
    def _prune_edges(self, edge_index: Tensor, x: Tensor) -> Tensor:
        row, col = edge_index
        mask = self.hp_filter(row, col, x, x)
        return edge_index[:, mask]


# =============================================================================
#  Training Loop
# =============================================================================

def train_one_run(config: Dict) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_data(config['dataset'])
    data = data.to(device)

    model = GATWithHashPipe(num_features=data.num_features,
                            num_classes=int(data.y.max().item() + 1),
                            hidden=config['hidden'],
                            num_layers=config['num_layers'],
                            heads=config['heads'],
                            use_hashpipe=config.get('model', 'hashpipe') == 'hashpipe',
                            top_k=config['T'],
                            cache_size=config['cache_size'],
                            sketch_dim=config['sketch'],
                            thresholds=tuple(config['tau']))

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scaler = GradScaler()

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        with autocast():
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # Evaluation
        model.eval()
        with torch.no_grad(), autocast():
            pred = out.argmax(dim=-1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

        if config.get('verbose', False):
            print(f"Epoch {epoch:02d} | Loss {loss.item():.4f} | Train {train_acc:.3f} | "
                  f"Val {val_acc:.3f} | Test {test_acc:.3f}")

    result = {
        'dataset': config['dataset'],
        'best_val_acc': best_val,
        'best_test_acc': best_test,
        'epochs': config['epochs']
    }
    _save_json(result, config['experiment_name'])
    return result


# =============================================================================
#  Utilities
# =============================================================================

def _save_json(obj: Dict, exp_name: str) -> None:
    out_dir = Path('.research') / 'iteration1'
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{exp_name}.json"
    with fp.open('w') as f:
        json.dump(obj, f, indent=2)
    print(json.dumps(obj, indent=2))
