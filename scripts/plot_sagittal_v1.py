import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


summary_path = Path("outputs/spider_sagittal_pipeline_v1/summary.json")
output_path = Path("outputs/spider_sagittal_pipeline_v1/overview.png")
summary = json.loads(summary_path.read_text())

level = summary["lumbar_level_end_to_end"]
combined = summary["detection_level_and_grade_end_to_end"]
grade = summary["pfirrmann_given_correct_detection_and_level"]
studies = summary["complete_studies"]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Sagittal Pipeline v1 — 38 validation studies", fontsize=17)

# 1. End-to-end results. These are counts out of lumbar references.
ax = axes[0, 0]
total = level["references"]
values = [
    total,
    level["correct_detection_and_level"],
    combined["all_three_correct"],
]
labels = [
    "Reference lumbar discs",
    "Correct detection + level",
    "Correct detection + level + grade",
]
bars = ax.barh(labels, values, color=["#94a3b8", "#3b82f6", "#16a34a"])
ax.invert_yaxis()
ax.set_xlim(0, total * 1.28)
ax.set_title("End-to-end results")
ax.set_xlabel("Discs")
for bar, value in zip(bars, values):
    ax.text(
        value + 2,
        bar.get_y() + bar.get_height() / 2,
        f"{value}/{total} ({value / total:.1%})",
        va="center",
    )

# 2. Pfirrmann confusion matrix, among 166 correctly detected and leveled discs.
ax = axes[0, 1]
matrix = np.asarray(grade["confusion_matrix"])
ax.imshow(matrix, cmap="Blues")
ax.set_title("Pfirrmann confusion matrix (n=166)")
ax.set_xlabel("Predicted grade")
ax.set_ylabel("Reference grade")
ax.set_xticks(range(5), range(1, 6))
ax.set_yticks(range(5), range(1, 6))
for row in range(5):
    for col in range(5):
        count = matrix[row, col]
        color = "white" if count > matrix.max() / 2 else "black"
        ax.text(col, row, str(count), ha="center", va="center", color=color)

# 3. Per-grade precision and recall.
ax = axes[1, 0]
grades = grade["per_grade"]
x = np.arange(5)
width = 0.36
ax.bar(x - width / 2, [g["precision"] for g in grades],
       width, label="Precision", color="#6366f1")
ax.bar(x + width / 2, [g["recall"] for g in grades],
       width, label="Recall", color="#f59e0b")
ax.set_xticks(x, [str(g["grade"]) for g in grades])
ax.set_ylim(0, 1)
ax.set_xlabel("Pfirrmann grade")
ax.set_ylabel("Score")
ax.set_title("Performance by grade")
ax.legend()

# 4. Complete five-level studies.
ax = axes[1, 1]
eligible = studies["eligible_with_five_reference_levels"]
counts = [
    studies["all_five_levels_correct"],
    studies["all_five_levels_and_grades_correct"],
]
bars = ax.bar(
    ["All five levels\ncorrect", "All five levels and\ngrades correct"],
    counts,
    color=["#3b82f6", "#16a34a"],
)
ax.set_ylim(0, eligible * 1.15)
ax.set_ylabel(f"Studies out of {eligible}")
ax.set_title("Complete studies")
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 0.5,
        f"{count}/{eligible}",
        ha="center",
    )

fig.text(
    0.5, 0.015,
    "Localization F1 = 0.9205 across 264 disc components  •  "
    "Grade macro F1 = 0.4061  •  Grade MAE = 0.6747",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved {output_path}")