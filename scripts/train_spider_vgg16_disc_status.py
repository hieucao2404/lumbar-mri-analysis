"""Frozen VGG16 comparison: herniation binary and three disc-status classes.

The encoder is shared across slices and remains ImageNet-pretrained/frozen.
Patient-separated OOF predictions calibrate herniation alert thresholds;
SPIDER validation never selects training settings or thresholds.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from scripts.train_spider_abnormal_disc_cnn import read_rows
from scripts.train_spider_abnormal_disc_vgg16 import check_manifests, encode_rows
from src.models.abnormal_disc_vgg16 import VGG16AbnormalDisc


CLASSES = ("normal", "bulging_only", "herniation")
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
SAMPLE_SHARE = {"normal": 0.25, "bulging_only": 0.60, "herniation": 0.15}


def patient_folds(rows: list[dict], count: int, seed: int) -> np.ndarray:
    labels = np.array([row["status"] == "herniation" for row in rows], dtype=int)
    patients = np.array([row["patient_id"] for row in rows])
    assignments = np.full(len(rows), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    for fold, (train, held_out) in enumerate(
        splitter.split(np.zeros(len(rows)), labels, patients)
    ):
        if not labels[train].any() or not labels[held_out].any():
            raise ValueError(f"Fold {fold} has no herniation positives")
        if set(patients[train]) & set(patients[held_out]):
            raise ValueError("Patient overlap across a fold")
        assignments[held_out] = fold
    if np.any(assignments < 0):
        raise ValueError("Unassigned disc")
    return assignments


def sample_weights(rows: list[dict]) -> torch.Tensor:
    counts = Counter((row["status"], row["patient_id"]) for row in rows)
    patients_per_class = Counter(group for group, _ in counts)
    if set(patients_per_class) != set(CLASSES):
        raise ValueError("Each training fold needs all three classes")
    return torch.tensor([
        SAMPLE_SHARE[row["status"]]
        / patients_per_class[row["status"]]
        / counts[(row["status"], row["patient_id"])]
        for row in rows
    ], dtype=torch.double)


def make_head(outputs: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.2), nn.Linear(512, outputs))


def train_head(model, features, rows, task, epochs, seed, device, label):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    outputs = 1 if task == "binary_herniation" else 3
    model.slice_head = make_head(outputs).to(device)
    targets = torch.tensor(
        [int(row["status"] == "herniation") if outputs == 1
         else CLASS_ID[row["status"]] for row in rows],
        dtype=torch.float32 if outputs == 1 else torch.long,
    )
    sampler = WeightedRandomSampler(
        sample_weights(rows), num_samples=len(rows), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(features, targets), batch_size=32, sampler=sampler
    )
    optimizer = torch.optim.AdamW(model.slice_head.parameters(), lr=1e-3,
                                   weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if outputs == 1 else nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.slice_head.train()
        running = 0.0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits_from_features(batch_features.to(device))
            loss = loss_fn(logits, batch_targets.to(device))
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        print(f"{label} epoch={epoch + 1}/{epochs} loss={running / len(loader):.4f}",
              flush=True)


@torch.no_grad()
def score_features(model, features, task, device):
    model.slice_head.eval()
    parts = []
    for start in range(0, len(features), 128):
        logits = model.logits_from_features(features[start:start + 128].to(device))
        probabilities = (logits.sigmoid() if task == "binary_herniation"
                         else logits.softmax(dim=1))
        parts.append(probabilities.cpu().numpy())
    return np.concatenate(parts)


def herniation_scores(probabilities: np.ndarray, task: str) -> np.ndarray:
    return (probabilities if task == "binary_herniation"
            else probabilities[:, CLASS_ID["herniation"]])


def write_predictions(path, rows, scores, task, folds=None):
    fields = ["patient_id", "series_name", "level", "status", "herniation",
              "herniation_score"]
    if task == "three_class":
        fields += [f"p_{name}" for name in CLASSES]
    if folds is not None:
        fields.append("fold")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            result = {key: row[key] for key in fields[:4]}
            result.update(herniation=int(row["status"] == "herniation"),
                          herniation_score=float(herniation_scores(scores, task)[index]))
            if task == "three_class":
                result.update({f"p_{name}": float(scores[index, class_id])
                               for class_id, name in enumerate(CLASSES)})
            if folds is not None:
                result["fold"] = int(folds[index])
            writer.writerow(result)


def choose_threshold(rows, probabilities, task, target=0.90):
    positive_scores = sorted(
        (float(score) for row, score in
         zip(rows, herniation_scores(probabilities, task))
         if row["status"] == "herniation"), reverse=True
    )
    if not positive_scores:
        raise ValueError("No herniation positives")
    return positive_scores[math.ceil(target * len(positive_scores)) - 1]


def summarize(rows, probabilities, task, threshold):
    truth = np.array([row["status"] == "herniation" for row in rows])
    scores = herniation_scores(probabilities, task)
    flags = scores >= threshold
    result = {
        "discs": len(rows), "herniations": int(truth.sum()),
        "herniation_prevalence": float(truth.mean()),
        "herniation_ap": float(average_precision_score(truth, scores)),
        "herniation_ap_among_abnormal": float(average_precision_score(
            [row["status"] == "herniation" for row in rows if row["status"] != "normal"],
            [score for row, score in zip(rows, scores) if row["status"] != "normal"],
        )),
        "tp": int((flags & truth).sum()), "fp": int((flags & ~truth).sum()),
        "fn": int((~flags & truth).sum()), "tn": int((~flags & ~truth).sum()),
        "bulging_false_alerts": int(sum(flag and row["status"] == "bulging_only"
                                         for flag, row in zip(flags, rows))),
    }
    result["sensitivity"] = result["tp"] / (result["tp"] + result["fn"])
    result["precision"] = (result["tp"] / (result["tp"] + result["fp"])
                           if result["tp"] + result["fp"] else 0.0)
    if task == "three_class":
        labels = np.array([CLASS_ID[row["status"]] for row in rows])
        predicted = probabilities.argmax(axis=1)
        result["class_order"] = list(CLASSES)
        result["confusion_matrix"] = confusion_matrix(
            labels, predicted, labels=list(range(3))
        ).tolist()
        result["macro_f1"] = float(f1_score(labels, predicted, average="macro"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.folds < 2:
        raise ValueError("Positive epochs and at least two folds required")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    training = read_rows(args.data / "training.csv")
    validation = read_rows(args.validation_manifest)
    check_manifests(training, validation)
    folds = patient_folds(training, args.folds, args.seed)

    # Freeze the same ImageNet VGG16 features for both task heads. Their
    # embeddings are cached only in memory, never fitted to validation labels.
    model = VGG16AbnormalDisc(pretrained=True).to(device)
    train_features = encode_rows(model, training, device)
    val_features = encode_rows(model, validation, device)
    args.output.mkdir(parents=True)
    summary = {}
    for task in ("binary_herniation", "three_class"):
        task_dir = args.output / task
        task_dir.mkdir()
        oof = np.zeros((len(training),) if task == "binary_herniation"
                       else (len(training), 3), dtype=np.float32)
        for fold in range(args.folds):
            train_indices = np.flatnonzero(folds != fold)
            held_indices = np.flatnonzero(folds == fold)
            fold_rows = [training[i] for i in train_indices]
            train_head(model, train_features[train_indices], fold_rows, task,
                       args.epochs, args.seed + fold, device,
                       f"{task} fold={fold + 1}/{args.folds}")
            oof[held_indices] = score_features(
                model, train_features[held_indices], task, device
            )
        threshold = choose_threshold(training, oof, task)
        write_predictions(task_dir / "oof_predictions.csv", training, oof, task, folds)

        train_head(model, train_features, training, task, args.epochs,
                   args.seed, device, f"{task} full")
        val_scores = score_features(model, val_features, task, device)
        write_predictions(task_dir / "validation_predictions.csv", validation,
                          val_scores, task)
        state = {key: value.detach().cpu()
                 for key, value in model.state_dict().items()}
        torch.save({
            "model_kind": f"spider_vgg16_{task}",
            "model_state_dict": state,
            "classes": list(CLASSES),
            "herniation_threshold": threshold,
            "sampling_shares": SAMPLE_SHARE,
            "encoder": "VGG16_Weights.IMAGENET1K_V1 features frozen",
            "epochs": args.epochs,
            "seed": args.seed,
        }, task_dir / "fixed.pt")
        report = {
            "task": task, "threshold_from_training_oof": threshold,
            "training_oof": summarize(training, oof, task, threshold),
            "reused_validation": summarize(validation, val_scores, task, threshold),
            "note": "SPIDER validation has been reused; exploratory only.",
        }
        (task_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        summary[task] = report
        print(json.dumps(report, indent=2), flush=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
