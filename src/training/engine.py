from collections.abc import Iterable

import torch
from torch import nn

from src.training.metrics import box_iou


def run_epoch(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    Run one training or evaluation epoch.

    Supplying an optimizer enable training. Without an optimizer,
    the model runs in evaluation mode.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_iou = 0.0
    sample_count = 0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(images)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]

        total_loss += loss.detach().item() * batch_size
        total_iou += box_iou(
            predictions.detach(),
            targets,
        ).sum().item()

        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("The data loader contained no samples")

    return {
        "loss": total_loss / sample_count,
        "iou": total_iou / sample_count,
    }