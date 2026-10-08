import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk


FIELDS = [
    "patient_id",
    "subset",
    "series_name",
    "image_path",
    "mask_path",
    "ivd_label",
    "mask_value",
    "sagittal_axis",
    "sagittal_slice",
    "x1",
    "y1",
    "x2",
    "y2",
    "pfirrmann_grade",
    "disc_bulging",
    "disc_herniation",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def build_manifest(root: Path) -> tuple[list[dict], list[str]]:
    overview = read_csv(root / "overview.csv")
    grading_rows = read_csv(root / "radiological_gradings.csv")

    grades_by_patient = defaultdict(list)
    for grade in grading_rows:
        grades_by_patient[grade["Patient"]].append(grade)

    rows = []
    skipped = []

    # First baseline: use standard sagittal T2 series.
    for series in overview:
        series_name = series["new_file_name"]

        if not series_name.endswith("_t2"):
            continue

        patient_id = series_name.split("_", maxsplit=1)[0]
        image_path = root / "images" / f"{series_name}.mha"
        mask_path = root / "masks" / f"{series_name}.mha"

        if not image_path.is_file() or not mask_path.is_file():
            skipped.append(f"{series_name}: image or mask missing")
            continue

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))

        image = sitk.DICOMOrient(image, "LPS")
        mask = sitk.DICOMOrient(mask, "LPS")

        if image.GetSize() != mask.GetSize():
            raise ValueError(f"{series_name}: image/mask sizes differ")

        if not np.allclose(image.GetSpacing(), mask.GetSpacing()):
            raise ValueError(f"{series_name}: image/mask spacing differs")

        if not np.allclose(image.GetOrigin(), mask.GetOrigin()):
            raise ValueError(f"{series_name}: image/mask origins differ")

        if not np.allclose(image.GetDirection(), mask.GetDirection()):
            raise ValueError(f"{series_name}: image/mask directions differ")

        # For this baseline, require the thick slice axis to be x.
        # SimpleITK array order is [z, y, x], so sagittal slices
        # are mask_array[:, :, sagittal_slice].
        spacing_x, spacing_y, spacing_z = image.GetSpacing()
        # SimpleITK spacing is ordered [x, y, z].
        # NumPy arrays from SimpleITK are ordered [z, y, x].
        spacing = np.asarray(image.GetSpacing())
        axis_xyz = int(np.argmax(spacing))
        sagittal_axis = 2 - axis_xyz

        # Confirm that the selected slice axis points approximately
        # left/right in physical patient coordinates (LPS).
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        slice_normal = direction[:, axis_xyz]

        if abs(slice_normal[0]) < 0.8:
            raise ValueError(
                f"{series_name}: thickest axis does not appear sagittal; "
                f"spacing={image.GetSpacing()}, direction={image.GetDirection()}"
            )
        # After LPS orientation, NumPy shape is [superior, posterior, left].
        # Flip the first array axis so superior appears at the top of a 2D image.
        mask_array = np.flip(
            sitk.GetArrayFromImage(mask),
            axis=0,
        )

        for grade in grades_by_patient[patient_id]:
            ivd_label = int(grade["IVD label"])
            mask_value = 200 + ivd_label

            if ivd_label < 1:
                skipped.append(
                    f"{series_name}: unsupported IVD label {ivd_label}"
                )
                continue

            other_axes = tuple(
                axis for axis in range(3) if axis != sagittal_axis
            )
            areas = np.count_nonzero(
                mask_array == mask_value,
                axis=other_axes,
            )
            sagittal_slice = int(np.argmax(areas))

            if areas[sagittal_slice] == 0:
                skipped.append(
                    f"{series_name}: mask value {mask_value} absent"
                )
                continue

            disc_slice = (
                np.take(mask_array, sagittal_slice, axis=sagittal_axis)
                == mask_value
            )
            pixel_rows, pixel_cols = np.nonzero(disc_slice)

            # Bounding box in [x1, y1, x2, y2) coordinates
            # on the 2D sagittal slice.
            x1 = int(pixel_cols.min())
            y1 = int(pixel_rows.min())
            x2 = int(pixel_cols.max()) + 1
            y2 = int(pixel_rows.max()) + 1

            rows.append(
                {
                    "patient_id": patient_id,
                    "subset": series["subset"],
                    "series_name": series_name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "ivd_label": ivd_label,
                    "mask_value": mask_value,
                    "sagittal_axis": sagittal_axis,
                    "sagittal_slice": sagittal_slice,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "pfirrmann_grade": int(grade["Pfirrman grade"]),
                    "disc_bulging": int(grade["Disc bulging"]),
                    "disc_herniation": int(grade["Disc herniation"]),
                }
            )

    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/spider"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/spider_discs.csv"),
    )
    args = parser.parse_args()

    rows, skipped = build_manifest(args.root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", args.output)
    print("Disc samples:", len(rows))
    print("Patients:", len({row["patient_id"] for row in rows}))
    print("Samples by subset:", Counter(row["subset"] for row in rows))
    print(
        "Pfirrmann grades:",
        Counter(row["pfirrmann_grade"] for row in rows),
    )
    print("Skipped:", len(skipped))
    for reason in skipped[:20]:
        print("  ", reason)


if __name__ == "__main__":
    main()