"""Command-line entry-point orchestrating smoke/full runs."""
import argparse, yaml, json
from pathlib import Path

import timm

from .train import SwiftAdapt, FlashAdapt, SourceOnly, Tent, CoTTA, OSHA
from .preprocess import build_loader
from .evaluate import run_stream, save_json, plot_accuracy, plot_latency

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config"
RES_DIR = ROOT / ".research" / "iteration1"
IMG_DIR = RES_DIR / "images"
RES_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------#

def _algo(name: str, model, las):
    return {
        "swiftadapt": lambda: SwiftAdapt(model, las_decay=las),
        "flashadapt": lambda: FlashAdapt(model, las_decay=las),
        "source": lambda: SourceOnly(model),
        "tent": lambda: Tent(model),
        "cotta": lambda: CoTTA(model),
        "osha": lambda: OSHA(model),
    }[name]()


# -----------------------------------------------------------------------------#

def run(cfg: dict):
    summary = {}
    for ds_name, ds_cfg in cfg["datasets"].items():
        for model_name in cfg["models"]:
            print(f"\n=== {ds_name} | {model_name} ===")
            bs_conf = cfg["batch_size"]
            bs = bs_conf[ds_name] if isinstance(bs_conf, dict) else bs_conf
            loader, _ = build_loader(
                hf_id=ds_cfg["hf_id"],
                split=ds_cfg["split"],
                take=ds_cfg["take"],
                batch_size=bs,
                model_name=model_name,
            )
            model = timm.create_model(model_name.split("/")[-1], pretrained=True).cuda().eval()
            for algo_name in ["source", "tent", "cotta", "osha", "flashadapt", "swiftadapt"]:
                algo = _algo(algo_name, model, cfg["las_decay"])
                metrics = run_stream(loader, algo)

                json_path = RES_DIR / f"{algo_name}_{ds_name}_{model_name.split('/')[-1]}.json"
                save_json(metrics, json_path.as_posix())
                plot_accuracy(metrics["acc"], (IMG_DIR / f"acc_{algo_name}_{ds_name}.pdf").as_posix())
                plot_latency(metrics["latency"], (IMG_DIR / f"lat_{algo_name}_{ds_name}.pdf").as_posix())

                print(f"Results for {algo_name.upper()} on {ds_name}:\n" + json.dumps(metrics, indent=2))
                summary[f"{algo_name}_{ds_name}"] = metrics
    return summary


# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--full-experiment", action="store_true")
    args = p.parse_args()
    assert args.smoke_test ^ args.full_experiment, "choose exactly one mode"

    cfg_file = CFG / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    run(cfg)