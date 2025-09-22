import os
from functools import lru_cache
from typing import Literal

import torch
from torch_geometric.datasets import Reddit, Planetoid
from torch_geometric.transforms import NormalizeFeatures

__all__ = ["load_data"]

data_root = os.getenv("DATA_ROOT", "./data")


@lru_cache(maxsize=None)
def load_data(dataset: Literal["Reddit", "Cora", "CiteSeer", "PubMed"]) -> torch_geometric.data.Data:  # type: ignore
    """Load and cache graph datasets with basic preprocessing."""
    name = dataset.lower()
    if name == "reddit":
        ds = Reddit(root=f"{data_root}/Reddit", transform=NormalizeFeatures())
    elif name in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root=f"{data_root}/{dataset}", name=dataset, transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    return ds[0]
