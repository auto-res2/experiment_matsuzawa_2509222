"""Main orchestration script.

This file provides an argument-driven CLI that supports the following calls:

    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment

Given that the original monolithic experiment script is unavailable, the
module focuses on configuration loading and graceful failure with clear
error messages.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .evaluate import evaluate_model
from .preprocess import prepare_datasets
from .train import train_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RESEARCH_DIR = PROJECT_ROOT / ".research" / "iteration1"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def _load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded config from %s", config_path)
        return config
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", config_path)
        raise e
    except yaml.YAMLError as e:
        logger.error("YAML parsing error in %s: %s", config_path, str(e))
        raise e


def _dump_json(obj: Dict[str, Any], filename: str) -> None:
    path = RESEARCH_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(json.dumps(obj, indent=2))  # echo to stdout for CI
    logger.info("Saved artefact to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APTA experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment")
    args = parser.parse_args()

    config_file = (
        CONFIG_DIR / "smoke_test.yaml" if args.smoke_test else CONFIG_DIR / "full_experiment.yaml"
    )
    config = _load_config(config_file)

    # Phase 1: preprocessing (if any)
    prepare_datasets(config)

    # Phase 2: training – expected to raise NotImplementedError until supplied
    try:
        train_summary = train_model(config)
    except NotImplementedError as e:
        logger.error(str(e))
        train_summary = {"status": "failed", "error": str(e)}

    _dump_json(train_summary, "train_summary.json")

    # Phase 3: evaluation – returns dummy result at present
    eval_summary = evaluate_model(config)
    _dump_json(eval_summary, "eval_summary.json")


if __name__ == "__main__":
    main()
