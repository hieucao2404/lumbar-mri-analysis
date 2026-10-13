from src.postprocess.disc_level_sequence import (
    LEVEL_NAMES,
    decode_lumbar_sequence,
)


def make_component(
    component_id: int,
    center_y: float,
    probabilities: list[float],
) -> dict:
    return {
        "component_id": component_id,
        "center_x": 120.0,
        "center_y": center_y,
        "x1": 105,
        "y1": round(center_y - 6),
        "x2": 135,
        "y2": round(center_y + 6),
        "area": 300,
        "level_probabilities": probabilities,
    }


def main() -> None:
    components = [
        # Upper discs
        make_component(
            1,
            40.0,
            [0.03, 0.02, 0.02, 0.01, 0.01, 0.91],
        ),
        make_component(
            2,
            70.0,
            [0.05, 0.03, 0.02, 0.01, 0.01, 0.88],
        ),

        # Lumbar discs
        make_component(
            3,
            100.0,
            [0.72, 0.12, 0.04, 0.02, 0.01, 0.09],
        ),
        make_component(
            4,
            130.0,
            [0.10, 0.70, 0.10, 0.02, 0.01, 0.07],
        ),
        make_component(
            5,
            160.0,
            [0.02, 0.14, 0.64, 0.12, 0.02, 0.06],
        ),
        make_component(
            6,
            190.0,
            [0.01, 0.02, 0.20, 0.58, 0.12, 0.07],
        ),
        make_component(
            7,
            220.0,
            [0.01, 0.01, 0.02, 0.15, 0.75, 0.06],
        ),
    ]

    result = decode_lumbar_sequence(
        components,
        learned_weight=1.0,
        position_weight=0.75,
        upper_weight=0.20,
    )

    print("Status:", result["status"])
    print("Reasons:", result["reasons"])
    print("Input components:", result["input_component_count"])
    print(
        "Selected indices:",
        result["selected_sorted_indices"],
    )
    print("Spacing CV:", round(result["spacing_cv"], 4))
    print(
        "Total score:",
        round(result["score"], 4),
    )

    print("\nAssignments:")

    predicted_levels = []

    for assignment in result["assignments"]:
        predicted_levels.append(assignment["level"])

        print(
            f"{assignment['level']:<6} "
            f"component={assignment['component_id']} "
            f"center_y={assignment['center_y']:.1f} "
            f"confidence={assignment['confidence']:.3f}"
        )

    assert result["status"] == "ok"
    assert predicted_levels == list(LEVEL_NAMES)
    assert result["selected_sorted_indices"] == [2, 3, 4, 5, 6]

    print("\nDisc-level sequence smoke test passed")


if __name__ == "__main__":
    main()