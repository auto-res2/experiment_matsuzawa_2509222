"""Model training related utilities."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Entry-point for training.

    Parameters
    ----------
    config : Dict[str, Any]
        A nested dictionary produced by `yaml.safe_load` on the
        corresponding experiment configuration file.

    Returns
    -------
    Dict[str, Any]
        A dictionary (JSON-serialisable) with the training summary that will
        be written to `.research/iteration1/` by `main.py`.
    """
    experiment_config = config.get("experiment", {})
    epochs = experiment_config.get("epochs", 1)
    batch_size = experiment_config.get("batch_size", 2)
    device_name = experiment_config.get("device", "cpu")
    dataset_name = experiment_config.get("dataset", "tiny-imagenet-c")
    num_workers = experiment_config.get("num_workers", 2)

    logger.info(f"Starting training with config: {experiment_config}")

    # Set up device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    # Prepare datasets and data loaders
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

    # Load dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Create model using timm
    model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    model = model.to(device)

    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_samples += len(data)

            # Log progress periodically
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Acc: {100. * correct / len(data):.2f}%')

        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * epoch_correct / epoch_samples

        logger.info(f'Epoch {epoch+1}/{epochs} completed - '
                   f'Avg Loss: {avg_loss:.4f}, '
                   f'Accuracy: {accuracy:.2f}%')

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    # Calculate final metrics
    training_time = time.time() - start_time
    final_loss = total_loss / (len(train_loader) * epochs)
    final_accuracy = 100. * total_correct / total_samples

    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final training accuracy: {final_accuracy:.2f}%")

    # Save the model
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "trained_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'final_accuracy': final_accuracy,
        'final_loss': final_loss,
    }, model_path)

    logger.info(f"Model saved to {model_path}")

    return {
        "status": "completed",
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "training_time_seconds": round(training_time, 2),
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_accuracy, 2),
        "model_path": str(model_path),
        "total_samples_processed": total_samples
    }
