import argparse
import sys
from pathlib import Path

import yaml

from .train import train_one_run
from .evaluate import evaluate

# ----------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg(fname: str):
    fp = CONFIG_DIR / fname
    try:
        with fp.open() as f:
            cfg = yaml.safe_load(f)
        return cfg
    except FileNotFoundError as e:
        print(f"Configuration file {fname} not found: {e}", file=sys.stderr)
        sys.exit(1)


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HASHPIPE-GAT experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run full benchmark experiment")
    args = parser.parse_args()

    if args.smoke_test:
        cfg = _load_cfg("smoke_test.yaml")
    else:
        cfg = _load_cfg("full_experiment.yaml")

    # ---------------- Run training --------------------------------------
    print("Starting training ...")
    train_res = train_one_run(cfg)

    # ---------------- Evaluation ---------------------------------------
    print("\nStarting evaluation ...")
    evaluate(cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()
