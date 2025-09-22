"""src/evaluate.py
Evaluation utilities – accuracy + simple JSON logging
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


__all__ = ["evaluate"]


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: Data, cfg: Dict[str, Any]) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1).cpu()
    acc_val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    acc_test = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    results = {
        "val_accuracy": acc_val,
        "test_accuracy": acc_test,
    }

    # -------- persistent research artefact ----------------------------------
    Path(".research/iteration1").mkdir(parents=True, exist_ok=True)
    out_file = Path(".research/iteration1/") / f"results_{cfg['run_name']}.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # always print to stdout for CI visibility
    print(json.dumps(results, indent=2))
    return results
