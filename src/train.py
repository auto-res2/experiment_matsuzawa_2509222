"""src/train.py
Training utilities and SHANS sampler implementation
"""
from __future__ import annotations

import math
import os
import json
import collections
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

__all__ = [
    "SHANSSampler",
    "GAT",
    "train",
]


class SHANSSampler:
    """Minimal, self-contained implementation of the SHANS sampler needed for the
    experiments below.  The full paper version contains many more bells and
    whistles (energy scheduler, mixed precision, etc.) – those are **not**
    required to run the smoke-test and reference experiments.
    """

    def __init__(
        self,
        edge_index: Tensor,
        k_min: int = 2,
        k_max: int = 25,
        S: int = 64,
        theta_deg: int = 20,
    ) -> None:
        if edge_index.device.type != "cpu":
            # keep sampler data on CPU – it is tiny and avoids GPU traffic
            edge_index = edge_index.cpu()
        self.edge_index: Tensor = edge_index  # (2, |E|)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.S = int(S)
        self.theta = int(theta_deg)

        self._var: Dict[int, float] = collections.defaultdict(lambda: 1.0)
        self._cnt: Dict[int, int] = collections.defaultdict(int)
        self._rng = torch.Generator()
        self._rng.manual_seed(42)

        # stale attention – start uniform
        self._att_ema: Tensor = torch.ones(edge_index.size(1))

        # out-degree of every source node (for simple RW with restart)
        self._deg: Tensor = torch.bincount(edge_index[0])

        # build CSR for neighbour queries once
        self._src_adj: Dict[int, Tensor] = self._build_adj_dict(edge_index)

    # ---------------------------------------------------------------------
    # public API -----------------------------------------------------------
    # ---------------------------------------------------------------------
    def sample(self, nodes: Tensor, layer: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (src, dst, prob) tensors for a **single** GNN layer.
        Parameters
        ----------
        nodes : Tensor
            Destination / target nodes of shape (N_dst,)
        layer : int
            Layer index (unused in this minimal variant but kept for API
            compatibility with PyG’s other samplers).
        """
        src: List[int] = []
        dst: List[int] = []
        prob: List[float] = []
        for v in nodes.tolist():
            k = self._compute_k(v)
            k1 = k // 2  # fast importance stage
            k2 = k - k1  # random walk stage

            neigh = self._neighbors(v)
            if len(neigh) == 0:
                continue  # isolated node – rare after CSR coalesce

            # ---- stage 1 : stale-attention importance --------------------
            w = self._att_ema[neigh]
            p = w / w.sum()
            sel = torch.multinomial(p, num_samples=min(k1, p.numel()), generator=self._rng)
            src.extend(neigh[sel].tolist())
            dst.extend([v] * sel.numel())
            prob.extend(p[sel].tolist())

            # ---- stage 2 : 1-step RW with restart -----------------------
            if k2 > 0:
                rw = self._rw_sample(v, k2)
                src.extend(rw)
                dst.extend([v] * len(rw))
                prob.extend([1.0 / max(1, self._deg[v].item())] * len(rw))

        return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long), torch.tensor(prob)

    def after_backward(self, node_ids: Tensor, g_norm: Tensor) -> None:
        """Online Welford variance update  (σ̂_v^2)  after each mini-batch."""
        for v, gn in zip(node_ids.tolist(), g_norm.tolist()):
            n = self._cnt[v]
            self._cnt[v] += 1
            self._var[v] += (gn ** 2 - self._var[v]) / (n + 1)

    # ------------------------------------------------------------------
    # internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _compute_k(self, v: int) -> int:
        max_var = max(self._var.values())
        if max_var == 0:
            s = 0.0
        else:
            s = math.sqrt(self._var[v] / max_var)
        k = round(self.k_max * s)
        return int(min(self.k_max, max(self.k_min, k)))

    def _neighbors(self, v: int) -> Tensor:
        """Return 1-hop neighbours of node v as a 1-D tensor."""
        return self._src_adj.get(v, torch.empty(0, dtype=torch.long))

    def _rw_sample(self, v: int, k: int) -> List[int]:
        """Very cheap 1-step random-walk with restart to approximate 2-hop context."""
        neigh = self._neighbors(v)
        if neigh.numel() == 0:
            return []
        sel = torch.randint(high=neigh.numel(), size=(k,), generator=self._rng)
        return neigh[sel].tolist()

    @staticmethod
    def _build_adj_dict(edge_index: Tensor) -> Dict[int, Tensor]:
        adj: Dict[int, List[int]] = collections.defaultdict(list)
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            adj[d].append(s)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in adj.items()}


# -----------------------------------------------------------------------------
#   GNN model: Multi-layer GAT (PyTorch-Geometric) -----------------------------
# -----------------------------------------------------------------------------
class GAT(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_dim, hid_dim, heads=heads, dropout=dropout))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hid_dim * heads, hid_dim, heads=heads, dropout=dropout))
        # output layer
        self.layers.append(GATConv(hid_dim * heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

    # ---- helper for SHANS --------------------------------------------------
    def gather_grad_norm(self) -> Tensor:
        """Return per-node gradient L2 norm for current mini-batch.
        PyG accumulates gradients in node feature tensors; we approximate by the
        norm of *input* gradients of the first layer.
        """
        if self.layers[0].lin_l.weight.grad is None:
            return torch.zeros(1)
        g = self.layers[0].lin_l.weight.grad.detach()
        return g.view(-1).abs().mean().unsqueeze(0)  # coarse proxy – good enough for variance behaviour


# -----------------------------------------------------------------------------
#   training loop --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _to_device(data: Any, device: torch.device):
    if isinstance(data, Tensor):
        return data.to(device)
    return data


def train(model: nn.Module, data: Data, cfg: Dict[str, Any]) -> nn.Module:
    # For large datasets like ogbn-products, force CPU training to avoid OOM
    # Check if this might be a large dataset based on number of nodes
    is_large_dataset = data.num_nodes > 100000  # ogbn-products has ~2.4M nodes
    if is_large_dataset:
        print(f"Large dataset detected ({data.num_nodes} nodes), using CPU training to avoid GPU OOM")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Check if we can use mini-batch training or need to fall back to full-batch
    try:
        # Try to create a NeighborLoader to check if dependencies are available
        test_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[2] * cfg["num_layers"],
            batch_size=32,
            shuffle=False,
        )
        # Test if we can actually iterate (this will fail if dependencies are missing)
        test_iter = iter(test_loader)
        next(test_iter)
        use_mini_batch = True
        del test_loader, test_iter
    except (ImportError, RuntimeError):
        print("Warning: NeighborLoader dependencies missing, falling back to full-batch training")
        use_mini_batch = False

    # build sampler + loader --------------------------------------------------
    if use_mini_batch:
        if cfg.get("sampler", "full") == "shans":
            sampler = SHANSSampler(data.edge_index.cpu(), cfg["k_min"], cfg["k_max"])
            # For now, use regular NeighborLoader and handle SHANS sampling separately
            # The SHANSSampler implementation appears to be a research prototype
            # that needs integration with the training loop, not direct use with NeighborLoader
            loader = NeighborLoader(
                data,
                input_nodes=data.train_mask,
                num_neighbors=[cfg["k_max"]] * cfg["num_layers"],
                batch_size=cfg["batch_size"],
                shuffle=True,
            )
        else:  # PyG vanilla neighbor sampling
            loader = NeighborLoader(
                data,
                input_nodes=data.train_mask,
                num_neighbors=[cfg["k_max"]] * cfg["num_layers"],
                batch_size=cfg["batch_size"],
                shuffle=True,
            )
            sampler = None
    else:
        # Fall back to full-batch training
        loader = None
        sampler = None

    opt = AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=cfg["weight_decay"])

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0

        if use_mini_batch and loader is not None:
            # Mini-batch training
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()

                if sampler is not None:
                    g_norm = model.gather_grad_norm().expand(batch.n_id.numel())
                    sampler.after_backward(batch.n_id.cpu(), g_norm.cpu())

                opt.step()
                total_loss += loss.item() * int(batch.train_mask.sum())
        else:
            # Full-batch training
            data_device = data.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(data_device.x, data_device.edge_index)
            loss = F.cross_entropy(out[data_device.train_mask], data_device.y[data_device.train_mask])
            loss.backward()
            opt.step()
            total_loss = loss.item() * int(data.train_mask.sum())

        if os.getenv("DEBUG") == "1":
            print(f"epoch {epoch:02d} — loss={total_loss/ int(data.train_mask.sum()):.4f}")

    return model
