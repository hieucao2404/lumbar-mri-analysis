from __future__ import annotations

from statistics import median


INFERIOR_TO_SUPERIOR_LEVELS = [
    "L5-S1",
    "L4-L5",
    "L3-L4",
    "L2-L3",
    "L1-L2",
]

OUTPUT_LEVEL_ORDER = [
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
]


def validate_component(
    component: dict,
    component_index: int,
) -> dict:
    required_fields = {
        "x1",
        "y1",
        "x2",
        "y2",
        "area",
        "center_x",
        "center_y",
    }

    missing_fields = required_fields - component.keys()

    if missing_fields:
        raise ValueError(
            f"Component {component_index} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    result = dict(component)
    result["component_index"] = component_index

    for key in required_fields:
        result[key] = float(result[key])

    if result["x2"] <= result["x1"]:
        raise ValueError(
            f"Component {component_index} has invalid x coordinates"
        )

    if result["y2"] <= result["y1"]:
        raise ValueError(
            f"Component {component_index} has invalid y coordinates"
        )

    if result["area"] <= 0:
        raise ValueError(
            f"Component {component_index} has invalid area"
        )

    return result


def assign_lumbar_levels(
    components: list[dict],
    minimum_area: int = 25,
    maximum_spacing_cv: float = 0.50,
) -> dict:
    """
    Assign the five lowest detected discs to lumbar levels.

    Assumptions:
    - The sagittal image has already been placed in the canonical,
      upright orientation.
    - Image y coordinates increase from top to bottom.
    - The lowest valid disc is L5-S1.
    - Additional components above the lumbar region may represent
      thoracic discs.

    This is a positional baseline. A sacrum or vertebra landmark model
    can later provide a stronger anatomical anchor.
    """

    validated_components = [
        validate_component(component, index)
        for index, component in enumerate(components)
    ]

    filtered_components = [
        component
        for component in validated_components
        if component["area"] >= minimum_area
    ]

    result = {
        "status": "uncertain",
        "reasons": [],
        "input_component_count": len(components),
        "filtered_component_count": len(filtered_components),
        "assignments": [],
    }

    if len(filtered_components) < 5:
        result["reasons"].append(
            "Fewer than five valid disc components were detected"
        )
        return result

    # In image coordinates, the largest center_y is the lowest disc.
    inferior_to_superior = sorted(
        filtered_components,
        key=lambda component: component["center_y"],
        reverse=True,
    )

    # SPIDER may contain thoracic discs above the lumbar region.
    # The lowest five discs are therefore selected.
    lumbar_components = inferior_to_superior[:5]

    vertical_gaps = []

    for inferior, superior in zip(
        lumbar_components,
        lumbar_components[1:],
    ):
        gap = inferior["center_y"] - superior["center_y"]

        if gap <= 0:
            result["reasons"].append(
                "Detected discs do not have a valid vertical order"
            )
            return result

        vertical_gaps.append(gap)

    median_gap = median(vertical_gaps)

    if median_gap <= 0:
        result["reasons"].append(
            "The median inter-disc spacing is invalid"
        )
        return result

    mean_gap = sum(vertical_gaps) / len(vertical_gaps)

    spacing_variance = sum(
        (gap - mean_gap) ** 2
        for gap in vertical_gaps
    ) / len(vertical_gaps)

    spacing_standard_deviation = spacing_variance ** 0.5
    spacing_cv = spacing_standard_deviation / mean_gap

    assignments = []

    for level, component in zip(
        INFERIOR_TO_SUPERIOR_LEVELS,
        lumbar_components,
    ):
        assignments.append({
            "position": level,
            "component_index": int(
                component["component_index"]
            ),
            "bounding_box": [
                round(component["x1"]),
                round(component["y1"]),
                round(component["x2"]),
                round(component["y2"]),
            ],
            "center_x": component["center_x"],
            "center_y": component["center_y"],
            "area": int(component["area"]),
        })

    # Final output should be presented from L1-L2 down to L5-S1.
    level_order = {
        level: index
        for index, level in enumerate(OUTPUT_LEVEL_ORDER)
    }

    assignments.sort(
        key=lambda assignment: level_order[
            assignment["position"]
        ]
    )

    result["assignments"] = assignments
    result["vertical_gaps"] = [
        round(gap, 3)
        for gap in vertical_gaps
    ]
    result["median_vertical_gap"] = round(median_gap, 3)
    result["spacing_cv"] = round(spacing_cv, 4)

    if len(filtered_components) > 9:
        result["reasons"].append(
            "More than nine disc components were detected; "
            "false-positive components may be present"
        )

    if spacing_cv > maximum_spacing_cv:
        result["reasons"].append(
            "The five selected discs have inconsistent vertical spacing"
        )

    if result["reasons"]:
        result["status"] = "uncertain"
    else:
        result["status"] = "ok"

    return result