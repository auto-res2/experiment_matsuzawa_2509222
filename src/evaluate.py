"""Evaluation, metrics & visualisation utilities."""
import json, os, time
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage,
)

# ---------------------------------------------------------------------------- #

def accuracy(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return (pred.argmax(1) == tgt).float().mean().item()


# ---------------------------------------------------------------------------- #

def run_stream(loader, algo, device="cuda", measure_energy=True):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    m = defaultdict(list)
    tic = time.time()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        t0 = time.time()
        with torch.no_grad():
            logits = algo.predict(x) if hasattr(algo, "predict") else algo(x)
        m["latency"].append(time.time() - t0)
        m["acc"].append(accuracy(logits, y))
        if measure_energy:
            m["power_w"].append(nvmlDeviceGetPowerUsage(handle) / 1000.0)
    m["elapsed"] = time.time() - tic
    m["mean_top1"] = float(np.mean(m["acc"]))
    m["fps"] = float(len(loader.dataset) / m["elapsed"])
    return m


# ---------------------------------------------------------------------------- #

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------- #

def plot_accuracy(acc: List[float], path: str):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(6, 3))
    plt.plot(acc)
    plt.xlabel("Batch index")
    plt.ylabel("Top-1 acc")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_latency(lat: List[float], path: str):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 3))
    plt.hist(np.array(lat) * 1e3, bins=30)
    plt.xlabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()