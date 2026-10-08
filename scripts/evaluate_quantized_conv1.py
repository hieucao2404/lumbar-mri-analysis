import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.data.lumbar_dataset import LumbarLocalizationDataset
from src.data.splits import create_split_indices
from src.models.lumbar_localizer import LumbarLocalizer
from src.quantization.conv2d import (
    UINT8_SCALE,
    conv2d_int32_reference,
    dequantize_conv_output,
    quantize_bias_int32,
    quantize_input_uint8,
    quantize_weights_int8_per_channel,
    calculate_requantization_multipliers,
    dequantize_activation_uint8,
    relu_max_pool2d_int32,
    requantize_accumulator_uint8,
    requantize_accumulator_uint8_fixed,
)
from src.quantization.config import(
    CONV1_OUTPUT_SCALE,
    CONV1_REQUANTIZATION_SHIFT,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare floating-point Conv1 against the "
            "integer reference implementation."
        )
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
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def create_error_metrics() -> dict[str, float | int]:
    return {
        "absolute_sum": 0.0,
        "squared_sum": 0.0,
        "maximum": 0.0,
        "count": 0,
    }


def update_error_metrics(
    metrics: dict[str, float | int],
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    error = expected - actual
    absolute_error = error.abs()

    metrics["absolute_sum"] += absolute_error.sum().item()
    metrics["squared_sum"] += error.square().sum().item()
    metrics["maximum"] = max(
        float(metrics["maximum"]),
        absolute_error.max().item(),
    )
    metrics["count"] += error.numel()


def print_error_metrics(
    name: str,
    metrics: dict[str, float | int],
) -> None:
    count = int(metrics["count"])

    mean_absolute_error = (
        float(metrics["absolute_sum"]) / count
    )

    root_mean_squared_error = math.sqrt(
        float(metrics["squared_sum"]) / count
    )

    print(name)
    print(f"  Mean absolute error: {mean_absolute_error:.8f}")
    print(f"  RMSE:                {root_mean_squared_error:.8f}")
    print(f"  Maximum error:       {float(metrics['maximum']):.8f}")


def required_signed_bits(value: int) -> int:
    """
    Return the number of signed bits needed to represent ±value.
    """
    return max(1, abs(value).bit_length() + 1)


def main() -> None:
    arguments = parse_arguments()

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

    validation_dataset = Subset(
        dataset,
        splits["validation"],
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

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

    integer_multipliers = calculate_requantization_multipliers(
        weight_scales,
        output_scale=CONV1_OUTPUT_SCALE,
        shift=CONV1_REQUANTIZATION_SHIFT,
    )

    fixed_output_metrics = create_error_metrics()

    total_output_values = 0
    different_output_values = 0
    one_lsb_differences = 0
    maximum_lsb_difference = 0
    saturated_output_values = 0

    conv_metrics = create_error_metrics()
    relu_metrics = create_error_metrics()
    pool_metrics = create_error_metrics()

    accumulator_minimum = None
    accumulator_maximum = None

    float_pooled_outputs: list[torch.Tensor] = []
    quantized_pooled_outputs: list[torch.Tensor] = []

    with torch.no_grad():
        for images, _ in validation_loader:
            float_conv = conv1(images)
            float_relu = F.relu(float_conv)
            float_pool = F.max_pool2d(
                float_relu,
                kernel_size=2,
            )

            input_uint8 = quantize_input_uint8(images)

            accumulator_int32 = conv2d_int32_reference(
                input_uint8,
                weights_int8,
                bias_int32,
                stride=1,
                padding=1,
            )

            pooled_accumulator = relu_max_pool2d_int32(
                accumulator_int32,
                kernel_size=2,
                stride=2,
            )

            float_requantized = requantize_accumulator_uint8(
                pooled_accumulator,
                weight_scales,
                output_scale=CONV1_OUTPUT_SCALE,
            )

            fixed_requantized = requantize_accumulator_uint8_fixed(
                pooled_accumulator,
                integer_multipliers,
                shift=CONV1_REQUANTIZATION_SHIFT,
            )

            lsb_difference = (
                float_requantized.to(torch.int16)
                - fixed_requantized.to(torch.int16)
            ).abs()

            total_output_values += lsb_difference.numel()

            different_output_values += (
                lsb_difference != 0
            ).sum().item()

            one_lsb_differences += (
                lsb_difference == 1
            ).sum().item()

            maximum_lsb_difference = max(
                maximum_lsb_difference,
                lsb_difference.max().item(),
            )

            fixed_dequantized = dequantize_activation_uint8(
                fixed_requantized,
                CONV1_OUTPUT_SCALE,
            )

            update_error_metrics(
                fixed_output_metrics,
                float_pool,
                fixed_dequantized,
            )

            quantized_conv = dequantize_conv_output(
                accumulator_int32,
                weight_scales,
                input_scale=UINT8_SCALE,
            )

            quantized_relu = F.relu(quantized_conv)
            quantized_pool = F.max_pool2d(
                quantized_relu,
                kernel_size=2,
            )

            float_pooled_outputs.append(
                float_pool.flatten().cpu()
            )

            quantized_pooled_outputs.append(
                quantized_pool.flatten().cpu()
            )

            update_error_metrics(
                conv_metrics,
                float_conv,
                quantized_conv,
            )

            update_error_metrics(
                relu_metrics,
                float_relu,
                quantized_relu,
            )

            update_error_metrics(
                pool_metrics,
                float_pool,
                quantized_pool,
            )

            current_minimum = accumulator_int32.min().item()
            current_maximum = accumulator_int32.max().item()

            if accumulator_minimum is None:
                accumulator_minimum = current_minimum
                accumulator_maximum = current_maximum
            else:
                accumulator_minimum = min(
                    accumulator_minimum,
                    current_minimum,
                )
                accumulator_maximum = max(
                    accumulator_maximum,
                    current_maximum,
                )

        all_float_pool = torch.cat(float_pooled_outputs)
        all_quantized_pool = torch.cat(quantized_pooled_outputs)

    # Conservative bound for any UINT8 input:
    #
    # |accumulator| <=
    #     255 × sum(|quantized weights|) + |quantized bias|
    weight_absolute_sums = (
        weights_int8
        .to(torch.int64)
        .abs()
        .flatten(start_dim=1)
        .sum(dim=1)
    )

    conservative_bounds = (
        255 * weight_absolute_sums
        + bias_int32.to(torch.int64).abs()
    )

    largest_conservative_bound = int(
        conservative_bounds.max().item()
    )

    observed_largest_magnitude = max(
        abs(int(accumulator_minimum)),
        abs(int(accumulator_maximum)),
    )

    print(f"Validation samples: {len(validation_dataset)}")
    print()
    print("Quantized parameter ranges")
    print(
        f"  Weights:      "
        f"{weights_int8.min().item()} to "
        f"{weights_int8.max().item()}"
    )
    print(
        f"  Weight scale: "
        f"{weight_scales.min().item():.10f} to "
        f"{weight_scales.max().item():.10f}"
    )
    print(
        f"  Bias:         "
        f"{bias_int32.min().item()} to "
        f"{bias_int32.max().item()}"
    )

    print()
    print("Accumulator analysis")
    print(
        f"  Observed range:       "
        f"{accumulator_minimum} to {accumulator_maximum}"
    )
    print(
        f"  Observed signed bits: "
        f"{required_signed_bits(observed_largest_magnitude)}"
    )
    print(
        f"  Conservative bound:   "
        f"+{largest_conservative_bound}"
    )
    print(
        f"  Conservative bits:    "
        f"{required_signed_bits(largest_conservative_bound)}"
    )
    print("  Selected RTL width:   32 bits")

    print()
    print_error_metrics(
        "Convolution output:",
        conv_metrics,
    )

    print()
    print_error_metrics(
        "After ReLU:",
        relu_metrics,
    )

    print()
    print_error_metrics(
        "After 2x2 max pooling:",
        pool_metrics,
    )

    exact_output_values = (
    total_output_values - different_output_values
    )

    exact_percentage = (
        100.0 * exact_output_values / total_output_values
    )

    different_percentage = (
        100.0 * different_output_values / total_output_values
    )

    saturation_percentage = (
        100.0 * saturated_output_values / total_output_values
    )

    print()
    print("Fixed-point requantization")
    print(
        f"  Integer multiplier range: "
        f"{integer_multipliers.min().item()} to "
        f"{integer_multipliers.max().item()}"
    )
    print(
        f"  Shared right shift:       "
        f"{CONV1_REQUANTIZATION_SHIFT}"
    )
    print(
        f"  Total output values:      "
        f"{total_output_values}"
    )
    print(
        f"  Exact float/fixed match:  "
        f"{exact_output_values}/{total_output_values} "
        f"({exact_percentage:.6f}%)"
    )
    print(
        f"  Different values:         "
        f"{different_output_values}/{total_output_values} "
        f"({different_percentage:.6f}%)"
    )
    print(
        f"  One-LSB differences:      "
        f"{one_lsb_differences}"
    )
    print(
        f"  Maximum LSB difference:   "
        f"{maximum_lsb_difference}"
    )
    print(
        f"  Saturated values:         "
        f"{saturated_output_values}/{total_output_values} "
        f"({saturation_percentage:.6f}%)"
    )

    print()
    print_error_metrics(
        "Complete fixed-point Conv1 block versus floating point:",
        fixed_output_metrics,
    )

    print()
    print("Output activation calibration")
    print(
        f"{'Threshold':>12} "
        f"{'Activation max':>16} "
        f"{'Output scale':>16} "
        f"{'Clipped':>12} "
        f"{'MAE':>14} "
        f"{'Max error':>14}"
    )

    percentiles = [
        ("99.00%", 0.9900),
        ("99.90%", 0.9990),
        ("99.99%", 0.9999),
        ("Maximum", 1.0000),
    ]

    for label, percentile in percentiles:
        if percentile == 1.0:
            threshold = all_quantized_pool.max()
        else:
            threshold = torch.quantile(
                all_quantized_pool,
                percentile,
            )

        output_scale = threshold / 255.0

        output_uint8 = torch.round(
            all_quantized_pool / output_scale
        ).clamp(0, 255).to(torch.uint8)

        reconstructed_output = (
            output_uint8.to(torch.float32)
            * output_scale
        )

        absolute_error = (
            all_float_pool - reconstructed_output
        ).abs()

        clipped_percentage = (
            (all_quantized_pool > threshold)
            .to(torch.float32)
            .mean()
            .item()
            * 100.0
        )

        print(
            f"{label:>12} "
            f"{threshold.item():16.8f} "
            f"{output_scale.item():16.10f} "
            f"{clipped_percentage:11.6f}% "
            f"{absolute_error.mean().item():14.8f} "
            f"{absolute_error.max().item():14.8f}"
        )


if __name__ == "__main__":
    main()