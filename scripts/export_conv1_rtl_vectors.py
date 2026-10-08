import argparse
import json
from pathlib import Path

import torch

from src.data.lumbar_dataset import LumbarLocalizationDataset
from src.data.splits import create_split_indices
from src.models.lumbar_localizer import LumbarLocalizer
from src.quantization.config import (
    CONV1_OUTPUT_SCALE,
    CONV1_REQUANTIZATION_SHIFT,
)
from src.quantization.conv2d import (
    UINT8_SCALE,
    calculate_requantization_multipliers,
    conv2d_int32_reference,
    quantize_bias_int32,
    quantize_input_uint8,
    quantize_weights_int8_per_channel,
    relu_max_pool2d_int32,
    requantize_accumulator_uint8_fixed,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Conv1 vectors for Verilog verification."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/lss_mri_dataset"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/lumbar_localizer_augmented/best_model.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/conv1_rtl_vectors/sample_000"),
    )
    parser.add_argument(
        "--validation-position",
        type=int,
        default=0,
        help="Position inside the validation split."
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def write_hex_tensor(
    path: Path,
    tensor: torch.Tensor,
    bits: int,
) -> None:
    """
    Write one tensor value per line as hexadecimal.

    Signed negative values are encoded using two's complement.
    Tensor values are flattened in PyTorch contiguous order.
    """
    hexadecimal_digits = (bits + 3) // 4
    mask = (1 << bits) - 1

    values = (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(-1)
        .tolist()
    )

    with path.open("w", encoding="utf-8") as file:
        for value in values:
            encoded_value = int(value) & mask

            file.write(
                f"{encoded_value:0{hexadecimal_digits}X}\n"
            )


def main() -> None:
    arguments = parse_arguments()

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location="cpu",
    )

    model = LumbarLocalizer()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    conv1 = model.features[0]

    dataset = LumbarLocalizationDataset(
        arguments.dataset_root,
        target_size=(128, 128),
    )

    splits = create_split_indices(
        number_of_samples=len(dataset),
        train_fraction=0.8,
        validation_fraction=0.1,
        seed=arguments.seed,
    )

    validation_indices = splits["validation"]

    if not (
        0
        <= arguments.validation_position
        < len(validation_indices)
    ):
        raise ValueError(
            "validation-position is outside the validation split"
        )

    dataset_index = validation_indices[
        arguments.validation_position
    ]

    image, _ = dataset[dataset_index]
    image = image.unsqueeze(0)

    input_uint8 = quantize_input_uint8(image)

    weights_int8, weight_scales = (
        quantize_weights_int8_per_channel(
            conv1.weight.detach()
        )
    )

    bias_int32 = quantize_bias_int32(
        conv1.bias.detach(),
        weight_scales,
        input_scale=UINT8_SCALE,
    )

    integer_multipliers = (
        calculate_requantization_multipliers(
            weight_scales,
            output_scale=CONV1_OUTPUT_SCALE,
            input_scale=UINT8_SCALE,
            shift=CONV1_REQUANTIZATION_SHIFT,
        )
    )

    accumulator_int32 = conv2d_int32_reference(
        input_uint8,
        weights_int8,
        bias_int32,
        stride=1,
        padding=1,
    )

    pooled_accumulator_int32 = relu_max_pool2d_int32(
        accumulator_int32,
        kernel_size=2,
        stride=2,
    )

    output_uint8 = requantize_accumulator_uint8_fixed(
        pooled_accumulator_int32,
        integer_multipliers,
        shift=CONV1_REQUANTIZATION_SHIFT,
    )

    write_hex_tensor(
        arguments.output_dir / "input_uint8.hex",
        input_uint8,
        bits=8,
    )

    write_hex_tensor(
        arguments.output_dir / "weights_int8.hex",
        weights_int8,
        bits=8,
    )

    write_hex_tensor(
        arguments.output_dir / "bias_int32.hex",
        bias_int32,
        bits=32,
    )

    write_hex_tensor(
        arguments.output_dir / "multipliers_int32.hex",
        integer_multipliers,
        bits=32,
    )

    write_hex_tensor(
        arguments.output_dir / "expected_accumulator_int32.hex",
        accumulator_int32,
        bits=32,
    )

    write_hex_tensor(
        arguments.output_dir
        / "expected_pooled_accumulator_int32.hex",
        pooled_accumulator_int32,
        bits=32,
    )

    write_hex_tensor(
        arguments.output_dir / "expected_output_uint8.hex",
        output_uint8,
        bits=8,
    )

    # Create a small trace for one central output pixel.
    trace_output_channel = 0
    trace_y = 64
    trace_x = 64

    input_patch = input_uint8[
        0,
        0,
        trace_y - 1:trace_y + 2,
        trace_x - 1:trace_x + 2,
    ].to(torch.int32)

    kernel = weights_int8[
        trace_output_channel,
        0,
    ].to(torch.int32)

    products = input_patch * kernel

    mac_without_bias = int(products.sum().item())
    trace_bias = int(
        bias_int32[trace_output_channel].item()
    )

    traced_accumulator = mac_without_bias + trace_bias

    expected_traced_accumulator = int(
        accumulator_int32[
            0,
            trace_output_channel,
            trace_y,
            trace_x,
        ].item()
    )

    if traced_accumulator != expected_traced_accumulator:
        raise RuntimeError(
            "Central-pixel trace does not match the full convolution"
        )

    trace = {
        "output_channel": trace_output_channel,
        "output_y": trace_y,
        "output_x": trace_x,
        "input_patch_uint8": input_patch.tolist(),
        "kernel_int8": kernel.tolist(),
        "products": products.tolist(),
        "mac_without_bias": mac_without_bias,
        "bias_int32": trace_bias,
        "accumulator_int32": traced_accumulator,
    }

    metadata = {
        "dataset_index": dataset_index,
        "validation_position": arguments.validation_position,
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_shape": list(input_uint8.shape),
        "weight_shape": list(weights_int8.shape),
        "bias_shape": list(bias_int32.shape),
        "accumulator_shape": list(accumulator_int32.shape),
        "pooled_accumulator_shape": list(
            pooled_accumulator_int32.shape
        ),
        "output_shape": list(output_uint8.shape),
        "input_format": "UINT8",
        "weight_format": "INT8 two's complement",
        "bias_format": "INT32 two's complement",
        "accumulator_format": "INT32 two's complement",
        "output_format": "UINT8",
        "input_scale": UINT8_SCALE,
        "output_scale": CONV1_OUTPUT_SCALE,
        "requantization_shift": CONV1_REQUANTIZATION_SHIFT,
        "input_layout": "NCHW; flattened C, Y, X",
        "weight_layout": "O, I, KY, KX",
        "output_layout": "NCHW; flattened C, Y, X",
        "hex_format": "one value per line",
    }

    with (
        arguments.output_dir / "metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (
        arguments.output_dir / "central_pixel_trace.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(trace, file, indent=2)

    print("Conv1 RTL vectors exported successfully.")
    print(f"Dataset index: {dataset_index}")
    print(f"Input shape: {tuple(input_uint8.shape)}")
    print(f"Weight shape: {tuple(weights_int8.shape)}")
    print(
        f"Accumulator shape: "
        f"{tuple(accumulator_int32.shape)}"
    )
    print(
        f"Pooled shape: "
        f"{tuple(pooled_accumulator_int32.shape)}"
    )
    print(f"Output shape: {tuple(output_uint8.shape)}")
    print(f"Output directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
