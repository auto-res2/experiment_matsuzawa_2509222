"""src/main.py
Command-line entry-point.  Usage:
  uv run python -m src.main --smoke-test
  uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import yaml

from .preprocess import load_dataset
from .train import GAT, train
from .evaluate import evaluate

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg(file_name: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / file_name, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _run(cfg: Dict[str, Any]):
    # reproducibility ---------------------------------------------------------
    import torch

    torch.manual_seed(cfg.get("seed", 42))

    data = load_dataset(cfg)

    model = GAT(
        in_dim=data.num_features,
        hid_dim=cfg["hidden_dim"],
        out_dim=int(data.y.max().item()) + 1,
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
    )

    trained_model = train(model, data, cfg)
    evaluate(trained_model, data, cfg)


def main():
    parser = argparse.ArgumentParser(description="SHANS experimental runner")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true", help="run quick CI sanity check")
    grp.add_argument("--full-experiment", action="store_true", help="run full-scale experiment")
    args = parser.parse_args()

    if args.smoke_test:
        cfg = _load_cfg("smoke_test.yaml")
        cfg["run_name"] = "smoke"
        _run(cfg)
    else:
        # 2-phase execution – smoke first, then full if smoke passes ----------
        smoke_cfg = _load_cfg("smoke_test.yaml")
        smoke_cfg["run_name"] = "prerun"
        _run(smoke_cfg)

        full_cfg = _load_cfg("full_experiment.yaml")
        full_cfg["run_name"] = "full"
        _run(full_cfg)


if __name__ == "__main__":
    # allow running via  `python src/main.py`  as well
    main()
