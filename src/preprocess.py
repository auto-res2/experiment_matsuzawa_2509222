"""src/preprocess.py
Dataset loading & minimal pre-processing shared by train / evaluate
"""
from __future__ import annotations

from typing import Dict, Any

import torch
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch_geometric.transforms import NormalizeFeatures, ToUndirected

__all__ = ["load_dataset"]


def load_dataset(cfg: Dict[str, Any]):
    name = cfg.get("dataset", "Cora")
    root = cfg.get("dataset_root", "./data")

    if name.lower() in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root, name, transform=NormalizeFeatures())
        data = ds[0]
        return data
    elif name.lower() == "ogbn-products":
        import builtins
        import torch
        from ogb.nodeproppred import PygNodePropPredDataset

        # Monkey patch input to automatically answer 'y' for non-interactive mode
        original_input = builtins.input
        def auto_confirm_input(prompt=""):
            print(f"{prompt}y (auto-confirmed)")
            return "y"
        builtins.input = auto_confirm_input

        # Monkey patch torch.load to handle weights_only issue
        original_torch_load = torch.load
        def patched_torch_load(f, *args, **kwargs):
            # Set weights_only=False for compatibility with OGB datasets
            kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)
        torch.load = patched_torch_load

        try:
            ds = PygNodePropPredDataset(root=root, name="ogbn-products"); data = ds[0]
        finally:
            # Restore original functions
            builtins.input = original_input
            torch.load = original_torch_load
        # masks ----------------------------------------------------------------
        split_idx = ds.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros_like(data.train_mask)
        data.test_mask = torch.zeros_like(data.train_mask)
        data.train_mask[split_idx["train"]] = True
        data.val_mask[split_idx["valid"]] = True
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze()
        data.edge_index = ToUndirected()(data).edge_index
        return data
    else:
        raise ValueError(f"Unsupported dataset {name}")
