import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.lumbar_dataset import LumbarLocalizationDataset
from src.data.splits import create_split_indices
from src.models.lumbar_localizer import LumbarLocalizer
from src.training.engine import run_epoch
from src.preprocessing.augmentation import RandomBoxTranslation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the whole-lumbar localization CNN."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/lss_mri_dataset"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lumbar_localizer"),
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    arguments = parse_arguments()
    set_random_seed(arguments.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # dataset = LumbarLocalizationDataset(
    #     arguments.dataset_root,
    #     target_size=(128, 128),
    # )

    evaluation_dataset = LumbarLocalizationDataset(
        arguments.dataset_root,
        target_size=(128, 128),
    )

    training_dataset = LumbarLocalizationDataset(
        arguments.dataset_root,
        target_size=(128, 128),
        augmentation=RandomBoxTranslation(
            horizontal_fraction=0.15,
            vertical_fraction=0.10,
            probability=1.0,
        ),
    )

    splits = create_split_indices(
        number_of_samples=len(evaluation_dataset),
        train_fraction=0.8,
        validation_fraction=0.1,
        seed=arguments.seed,
    )

    train_dataset = Subset(training_dataset, splits["train"])
    validation_dataset = Subset(
        evaluation_dataset,
        splits["validation"],
    )

    loader_generator = torch.Generator().manual_seed(
        arguments.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = LumbarLocalizer().to(device)
    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arguments.learning_rate,
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = arguments.output_dir / "best_model.pt"
    history_path = arguments.output_dir / "history.json"

    best_validation_iou = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "validation_loss": validation_metrics["loss"],
            "validation_iou": validation_metrics["iou"],
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:03d}/{arguments.epochs:03d} | "
            f"train loss {train_metrics['loss']:.6f} | "
            f"train IoU {train_metrics['iou']:.4f} | "
            f"validation loss {validation_metrics['loss']:.6f} | "
            f"validation IoU {validation_metrics['iou']:.4f}"
        )

        if validation_metrics["iou"] > best_validation_iou:
            best_validation_iou = validation_metrics["iou"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_metrics["loss"],
                    "validation_iou": validation_metrics["iou"],
                    "target_size": (128, 128),
                    "seed": arguments.seed,
                },
                checkpoint_path,
            )

            print(
                f"  Saved new best model: "
                f"IoU={best_validation_iou:.4f}"
            )

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    print("\nTraining completed.")
    print(f"Best validation IoU: {best_validation_iou:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print("The test set remains untouched for final evaluation.")


if __name__ == "__main__":
    main()