"""Replay one guarded SPIDER scan and visualize attention-VGG16 decisions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from src.inference.herniation_levels import (
    LEVELS,
    LOCALIZATION_CHECKPOINT,
    level_bands,
    load_model,
    load_series_volume,
    localize_levels,
    make_stack,
    read_localization_records,
)
from src.inference.mri_slice_selector import select_sagittal_slice
from src.models.attention_vgg16_disc import AttentionVGG16Disc


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-name", default="31_t2")
    parser.add_argument("--focus-level", default="L4-L5", choices=LEVELS)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/processed/spider_bulge_herniation_mri_only_guarded_v2"),
    )
    parser.add_argument(
        "--experiment", type=Path,
        default=Path("outputs/spider_abnormal_vgg16_attention_guarded_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)

    rows = [row for row in read_csv(args.data / "validation_evaluable.csv")
            if row["series_name"] == args.series_name]
    if not rows:
        raise ValueError("Requested series is not in evaluable validation")
    rows.sort(key=lambda row: LEVELS.index(row["level"]))
    if args.focus_level not in {row["level"] for row in rows}:
        raise ValueError("Focus level is unavailable in this scan")
    saved = {row["level"]: row for row in read_csv(
        args.experiment / "validation_predictions.csv")
        if row["series_name"] == args.series_name}
    calibration = json.loads((args.experiment / "calibration.json").read_text())
    threshold = float(calibration["threshold"])

    record = read_localization_records()[args.series_name]
    volume = load_series_volume(args.series_name)
    localizer = load_model(LOCALIZATION_CHECKPOINT, device)
    selected, selected_image, selection = select_sagittal_slice(
        record, volume, localizer, device, middle_window=2,
        selection_mode="guarded_confidence",
    )
    boxes, decoder = localize_levels(
        record, volume, localizer, device,
        localization_image=selected_image,
    )
    bands = level_bands(boxes, volume.shape[0])
    audit = next(row for row in read_csv(args.data / "slice_audit.csv")
                 if row["series_name"] == args.series_name)
    if selected != int(audit["selected_slice"]):
        raise ValueError("Replayed slice differs from the saved crop audit")

    stacks = []
    for row in rows:
        level = row["level"]
        if level not in boxes:
            raise ValueError(f"No replayed box for {level}")
        saved_stack = np.load(row["stack_path"], allow_pickle=False)
        replayed = make_stack(
            volume, selected, boxes[level]["box"],
            crop_scale=1.5, posterior_shift=0.25, band=bands[level],
        )
        if not np.array_equal(saved_stack, replayed):
            raise ValueError(f"Replayed crop differs from saved input: {level}")
        stacks.append(saved_stack)

    checkpoint = torch.load(
        args.experiment / "fixed.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("model_kind") != "abnormal_disc_vgg16_slice_attention":
        raise ValueError("Checkpoint is not the binary attention VGG16")
    model = AttentionVGG16Disc(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inputs = torch.from_numpy(np.stack(stacks).astype(np.float32) / 255.0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        features = model.encode(inputs)
        probabilities = model.logits_from_features(features).sigmoid().float()
        attention = torch.softmax(
            model.slice_attention(features).squeeze(-1).float(), dim=1
        )
    probabilities = probabilities.cpu().numpy()
    attention = attention.cpu().numpy()
    for index, row in enumerate(rows):
        if row["level"] not in saved:
            raise ValueError(f"Missing saved prediction for {row['level']}")
        difference = abs(probabilities[index] - float(saved[row["level"]]["abnormal_score"]))
        if difference > 1e-3:
            raise ValueError(f"Checkpoint score differs from saved score: {row['level']}")

    display_slice = volume[:, :, selected]
    positive = display_slice[display_slice > 0]
    low, high = np.percentile(positive, [1, 99]) if positive.size else (0, 1)
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 8, height_ratios=[3.2, 1.6, 1.0])
    ax_mri = fig.add_subplot(grid[0, :5])
    ax_mri.imshow(display_slice, cmap="gray", vmin=low, vmax=high)
    ax_mri.set_title(
        f"{args.series_name}: original sagittal T2, selected slice {selected}\n"
        "Boxes mark localized discs, not lesion outlines",
        fontsize=13,
    )
    ax_mri.axis("off")
    details = []
    for index, row in enumerate(rows):
        level = row["level"]
        score = float(probabilities[index])
        flag = score >= threshold
        label_abnormal = row["status"] != "normal"
        color = ("#d62728" if flag else "#2b6cb0") if not label_abnormal else (
            "#2ca02c" if flag else "#e69f00")
        x1, y1, x2, y2 = boxes[level]["box"]
        ax_mri.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor=color, linewidth=2.2))
        ax_mri.text(x1, max(0, y1 - 3), level, color="white", fontsize=9,
                    fontweight="bold", bbox={"facecolor": color, "pad": 1,
                                              "edgecolor": "none", "alpha": 0.85})
        details.append({
            "level": level, "original_status": row["status"],
            "abnormal_score": score, "review_flag": bool(flag),
            "disc_box_original": boxes[level]["box"],
            "slice_attention_weights": attention[index].tolist(),
        })

    ax_text = fig.add_subplot(grid[0, 5:])
    ax_text.axis("off")
    lines = ["Level     Original label    Score   Review", "─" * 43]
    for item in details:
        lines.append(f"{item['level']:<9} {item['original_status']:<16} "
                     f"{item['abnormal_score']:>5.3f}   "
                     f"{'FLAG' if item['review_flag'] else 'clear'}")
    lines += ["", f"Training OOF threshold: {threshold:.3f}",
              "Green = true abnormal flag",
              "Red = false flag on a normal disc",
              "Blue = correctly unflagged normal disc",
              "Orange = missed abnormal disc", "",
              "Illustrative reused-validation case.",
              "A flag does not identify the subtype."]
    ax_text.text(0, 0.98, "\n".join(lines), va="top", ha="left",
                 fontsize=10, family="monospace", transform=ax_text.transAxes)

    focus_index = next(index for index, row in enumerate(rows)
                       if row["level"] == args.focus_level)
    focus_stack = stacks[focus_index]
    for slice_index in range(8):
        ax = fig.add_subplot(grid[1, slice_index])
        ax.imshow(focus_stack[slice_index, 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{selected + slice_index - 3}\nw={attention[focus_index, slice_index]:.2f}",
                     fontsize=9)
        ax.axis("off")
    ax_bar = fig.add_subplot(grid[2, :])
    x = np.arange(8)
    ax_bar.bar(x, attention[focus_index], color="#1764a5")
    ax_bar.set_xticks(x, [str(selected + index - 3) for index in x])
    ax_bar.set(xlabel="Sagittal slice index", ylabel="Attention weight",
               title=f"{args.focus_level}: attention across the eight input crops")
    ax_bar.set_ylim(0, max(0.2, float(attention[focus_index].max()) * 1.2))
    ax_bar.grid(axis="y", alpha=0.2)
    fig.suptitle("Attention VGG16 abnormal-disc review example", fontsize=17,
                 fontweight="bold")

    args.output_dir.mkdir(parents=True)
    image_path = args.output_dir / "mri_review.png"
    fig.savefig(image_path, dpi=180, facecolor="white")
    plt.close(fig)
    summary = {
        "series_name": args.series_name,
        "selected_slice": selected,
        "slice_selection": selection["method"],
        "decoder_source": decoder["source"],
        "threshold": threshold,
        "focus_level": args.focus_level,
        "levels": details,
        "image": str(image_path),
        "note": "Disc boxes and binary abnormal flags; not lesion localization or subtype diagnosis.",
    }
    (args.output_dir / "sample_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
