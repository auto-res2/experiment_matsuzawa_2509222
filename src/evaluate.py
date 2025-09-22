"""Evaluation and analysis utilities."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation and print results to stdout.

    This function loads a trained model and evaluates it on the test dataset.
    """
    experiment_config = config.get("experiment", {})
    batch_size = experiment_config.get("batch_size", 2)
    device_name = experiment_config.get("device", "cpu")
    dataset_name = experiment_config.get("dataset", "tiny-imagenet-c")
    num_workers = experiment_config.get("num_workers", 2)

    logger.info(f"Starting evaluation with config: {experiment_config}")

    # Set up device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    # Check if trained model exists
    models_dir = Path(__file__).resolve().parents[1] / "models"
    model_path = models_dir / "trained_model.pth"

    if not model_path.exists():
        logger.warning("No trained model found, returning dummy evaluation results")
        dummy_result = {"status": "skipped", "reason": "no trained model found"}
        print(json.dumps(dummy_result, indent=2), file=sys.stdout)
        return dummy_result

    # Prepare test dataset
    data_dir = Path(__file__).resolve().parents[1] / "data"

    if "tiny-imagenet" in dataset_name.lower():
        input_size = 64
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        input_size = 224
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Load the trained model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Evaluation loop
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    logger.info("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += len(data)

            # Log progress periodically
            if batch_idx % 100 == 0:
                logger.info(f'Test Batch {batch_idx}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Acc: {100. * correct / len(data):.2f}%')

    # Calculate final metrics
    evaluation_time = time.time() - start_time
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * total_correct / total_samples

    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    logger.info(f"Test accuracy: {accuracy:.2f}%")
    logger.info(f"Test loss: {avg_loss:.4f}")

    result = {
        "status": "completed",
        "test_accuracy": round(accuracy, 2),
        "test_loss": round(avg_loss, 4),
        "evaluation_time_seconds": round(evaluation_time, 2),
        "total_test_samples": total_samples,
        "model_path": str(model_path)
    }

    # Always print to stdout for CI visibility
    print(json.dumps(result, indent=2), file=sys.stdout)
    return result
