from src.postprocess.disc_level_assignment import (
    assign_lumbar_levels,
)


def make_component(
    center_x: float,
    center_y: float,
    width: int = 30,
    height: int = 12,
) -> dict:
    x1 = round(center_x - width / 2)
    x2 = round(center_x + width / 2)
    y1 = round(center_y - height / 2)
    y2 = round(center_y + height / 2)

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area": width * height,
        "center_x": center_x,
        "center_y": center_y,
    }


def main() -> None:
    # Seven visible discs:
    # five lumbar discs plus two discs above L1-L2.
    components = [
        make_component(105, 70),
        make_component(103, 220),
        make_component(102, 130),
        make_component(104, 40),
        make_component(101, 190),
        make_component(103, 100),
        make_component(101, 160),
    ]

    result = assign_lumbar_levels(components)

    print("Status:", result["status"])
    print("Reasons:", result["reasons"])
    print(
        "Input components:",
        result["input_component_count"],
    )
    print(
        "Filtered components:",
        result["filtered_component_count"],
    )
    print("Vertical gaps:", result["vertical_gaps"])
    print("Spacing CV:", result["spacing_cv"])

    print()
    print("Assigned lumbar levels:")

    for assignment in result["assignments"]:
        print(
            assignment["position"],
            "center_y=",
            assignment["center_y"],
            "box=",
            assignment["bounding_box"],
        )

    positions = [
        assignment["position"]
        for assignment in result["assignments"]
    ]

    assert positions == [
        "L1-L2",
        "L2-L3",
        "L3-L4",
        "L4-L5",
        "L5-S1",
    ]

    centers = {
        assignment["position"]: assignment["center_y"]
        for assignment in result["assignments"]
    }

    assert centers["L5-S1"] == 220
    assert centers["L4-L5"] == 190
    assert centers["L3-L4"] == 160
    assert centers["L2-L3"] == 130
    assert centers["L1-L2"] == 100

    incomplete_result = assign_lumbar_levels(
        components[:4]
    )

    assert incomplete_result["status"] == "uncertain"
    assert incomplete_result["assignments"] == []

    print()
    print("Disc-level assignment smoke test passed")


if __name__ == "__main__":
    main()