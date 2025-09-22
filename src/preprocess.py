"""Dataset download & DataLoader helper."""
import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from timm.data import resolve_model_data_config, create_transform


# ---------------------------------------------------------------------------- #

def build_loader(
    *,
    hf_id: str,
    split: str,
    take: int | None,
    batch_size: int,
    model_name: str,
    num_workers: int = 8,
    shuffle: bool = False,
) -> Tuple[DataLoader, Dict[str, Any]]:
    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    if take:
        ds = ds.select(range(take))

    # model-specific preprocessing
    import timm

    dummy_model = timm.create_model(model_name.split("/")[-1], pretrained=False)
    cfg = resolve_model_data_config(dummy_model)
    tfm = create_transform(**cfg, is_training=False)

    def _tfm(b):
        b["pixel_values"] = tfm(b["image"])
        return b

    ds = ds.with_transform(_tfm)

    def _collate(batch):
        imgs = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return imgs, labels

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, {"num_classes": len(set(ds["label"]))}