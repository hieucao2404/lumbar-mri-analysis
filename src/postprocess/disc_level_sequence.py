from __future__ import annotations

from itertools import combinations
from math import log
from typing import Sequence

import numpy as np


LEVEL_NAMES = (
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
)

UPPER_CLASS_ID = 5
NUMBER_OF_CLASSES = 6
EPSILON = 1e-8


def _validate_component(
    component: dict,
    original_index: int,
) -> dict:
    if "center_y" not in component:
        raise ValueError(
            f"Component {original_index} has no center_y"
        )

    if "level_probabilities" not in component:
        raise ValueError(
            f"Component {original_index} has no "
            "level_probabilities"
        )

    center_y = float(component["center_y"])

    probabilities = np.asarray(
        component["level_probabilities"],
        dtype=np.float64,
    )

    if probabilities.shape != (NUMBER_OF_CLASSES,):
        raise ValueError(
            f"Component {original_index}: expected six level "
            f"probabilities, got shape {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"Component {original_index} has non-finite "
            "probabilities"
        )

    if np.any(probabilities < 0):
        raise ValueError(
            f"Component {original_index} has negative probabilities"
        )

    probability_sum = float(probabilities.sum())

    if probability_sum <= 0:
        raise ValueError(
            f"Component {original_index} has zero probability sum"
        )

    # Normalize again in case the provided values contain small
    # numerical errors.
    probabilities = probabilities / probability_sum

    return {
        **component,
        "original_index": original_index,
        "center_y": center_y,
        "level_probabilities": probabilities,
    }


def _spacing_cv(selected_components: Sequence[dict]) -> float:
    centers = np.asarray(
        [
            component["center_y"]
            for component in selected_components
        ],
        dtype=np.float64,
    )

    gaps = np.diff(centers)

    if len(gaps) == 0:
        return 0.0

    mean_gap = float(gaps.mean())

    if mean_gap <= 0:
        return float("inf")

    return float(gaps.std() / mean_gap)


def decode_lumbar_sequence(
    components: list[dict],
    learned_weight: float = 1.0,
    position_weight: float = 0.75,
    upper_weight: float = 0.20,
    maximum_spacing_cv: float = 0.55,
) -> dict:
    """
    Select and label five lumbar discs from detected components.

    Required fields for each component:

        {
            "center_y": float,
            "level_probabilities": [
                p_L1_L2,
                p_L2_L3,
                p_L3_L4,
                p_L4_L5,
                p_L5_S1,
                p_upper,
            ],
        }

    The image must have canonical upright orientation, with increasing
    y coordinates moving from superior to inferior.

    The decoder combines three sources of information:

    1. Learned level probabilities.
    2. The expectation that the five lowest valid discs are lumbar.
    3. The anatomical order L1-L2 through L5-S1.

    Five components are selected in increasing vertical order. The
    first selected component becomes L1-L2 and the last becomes L5-S1.
    """

    if learned_weight < 0:
        raise ValueError("learned_weight must be non-negative")

    if position_weight < 0:
        raise ValueError("position_weight must be non-negative")

    if upper_weight < 0:
        raise ValueError("upper_weight must be non-negative")

    validated = [
        _validate_component(component, index)
        for index, component in enumerate(components)
    ]

    ordered = sorted(
        validated,
        key=lambda component: component["center_y"],
    )

    result = {
        "status": "uncertain",
        "reasons": [],
        "input_component_count": len(components),
        "assignments": [],
        "unselected_components": [],
    }

    if len(ordered) < 5:
        result["reasons"].append(
            "Fewer than five disc components were detected"
        )
        return result

    component_count = len(ordered)

    # The positional baseline expects the lumbar discs to occupy the
    # five lowest component positions.
    expected_indices = np.arange(
        component_count - 5,
        component_count,
        dtype=np.float64,
    )

    best_score = -float("inf")
    best_indices: tuple[int, ...] | None = None
    best_scores: dict | None = None

    for candidate_indices in combinations(
        range(component_count),
        5,
    ):
        learned_score = 0.0

        for level_id, component_index in enumerate(
            candidate_indices
        ):
            probability = ordered[
                component_index
            ]["level_probabilities"][level_id]

            learned_score += log(
                max(float(probability), EPSILON)
            )

        candidate_array = np.asarray(
            candidate_indices,
            dtype=np.float64,
        )

        # A value of zero means the selected discs are exactly the
        # five lowest detected components.
        position_distance = float(
            np.abs(candidate_array - expected_indices).sum()
        )

        position_score = -position_distance

        # Components above L1-L2 should normally represent upper
        # thoracic or thoracolumbar discs.
        upper_score = 0.0

        first_lumbar_index = candidate_indices[0]

        for component_index in range(first_lumbar_index):
            upper_probability = ordered[
                component_index
            ]["level_probabilities"][UPPER_CLASS_ID]

            upper_score += log(
                max(float(upper_probability), EPSILON)
            )

        total_score = (
            learned_weight * learned_score
            + position_weight * position_score
            + upper_weight * upper_score
        )

        if total_score > best_score:
            best_score = total_score
            best_indices = candidate_indices
            best_scores = {
                "learned_score": learned_score,
                "position_score": position_score,
                "upper_score": upper_score,
                "total_score": total_score,
            }

    if best_indices is None or best_scores is None:
        raise RuntimeError(
            "Sequence decoder could not produce an assignment"
        )

    selected_index_set = set(best_indices)
    selected_components = []

    for level_id, sorted_index in enumerate(best_indices):
        component = ordered[sorted_index]
        probabilities = component["level_probabilities"]

        assignment = {
            "level_id": level_id,
            "level": LEVEL_NAMES[level_id],
            "original_index": component["original_index"],
            "sorted_index": sorted_index,
            "center_y": component["center_y"],
            "confidence": float(probabilities[level_id]),
            "level_probabilities": probabilities.tolist(),
        }

        # Preserve useful component fields such as component_id,
        # bounding box, area, and center_x.
        for key in (
            "component_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "area",
            "center_x",
        ):
            if key in component:
                assignment[key] = component[key]

        result["assignments"].append(assignment)
        selected_components.append(component)

    for sorted_index, component in enumerate(ordered):
        if sorted_index not in selected_index_set:
            result["unselected_components"].append({
                "original_index": component["original_index"],
                "sorted_index": sorted_index,
                "center_y": component["center_y"],
                "upper_probability": float(
                    component[
                        "level_probabilities"
                    ][UPPER_CLASS_ID]
                ),
            })

    spacing_cv = _spacing_cv(selected_components)

    result["spacing_cv"] = spacing_cv
    result["score"] = best_score
    result["score_components"] = best_scores
    result["selected_sorted_indices"] = list(best_indices)

    if spacing_cv > maximum_spacing_cv:
        result["reasons"].append(
            "Selected lumbar discs have inconsistent vertical spacing"
        )

    if component_count > 10:
        result["reasons"].append(
            "More than ten disc components were detected"
        )

    if result["reasons"]:
        result["status"] = "uncertain"
    else:
        result["status"] = "ok"

    return result