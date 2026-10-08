import torch
from torch import nn

from src.quantization.conv2d import (
    UINT8_SCALE,
    conv2d_int32_reference,
    dequantize_conv_output,
    quantize_bias_int32,
    quantize_input_uint8,
    quantize_weights_int8_per_channel,
    calculate_requantization_multipliers,
    requantize_accumulator_uint8_fixed,
)

from src.quantization.config import (
    CONV1_OUTPUT_SCALE,
    CONV1_REQUANTIZATION_SHIFT,
)
from src.quantization.conv2d import (
    dequantize_activation_uint8,
    relu_max_pool2d_int32,
    requantize_accumulator_uint8,
)


def test_uint8_input_quantization() -> None:
    input_tensor = torch.tensor(
        [0.0, 0.5, 1.0],
        dtype=torch.float32,
    )

    quantized = quantize_input_uint8(input_tensor)

    assert quantized.dtype == torch.uint8
    assert torch.equal(
        quantized,
        torch.tensor([0, 128, 255], dtype=torch.uint8),
    )


def test_integer_convolution_known_values() -> None:
    input_uint8 = torch.tensor(
        [[[[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]]],
        dtype=torch.uint8,
    )

    weights_int8 = torch.ones(
        (1, 1, 3, 3),
        dtype=torch.int8,
    )

    bias_int32 = torch.tensor(
        [10],
        dtype=torch.int32,
    )

    output = conv2d_int32_reference(
        input_uint8,
        weights_int8,
        bias_int32,
        padding=0,
    )

    # Sum from 1 through 9 is 45, plus bias 10.
    assert output.shape == (1, 1, 1, 1)
    assert output.item() == 55


def test_quantized_convolution_approximates_float() -> None:
    torch.manual_seed(42)

    convolution = nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        padding=1,
    )

    input_float = torch.randint(
        0,
        256,
        (1, 1, 16, 16),
        dtype=torch.int32,
    ).to(torch.float32) / 255.0

    with torch.no_grad():
        expected_float = convolution(input_float)

    input_uint8 = quantize_input_uint8(input_float)

    weights_int8, weight_scales = (
        quantize_weights_int8_per_channel(
            convolution.weight.detach()
        )
    )

    bias_int32 = quantize_bias_int32(
        convolution.bias.detach(),
        weight_scales,
        input_scale=UINT8_SCALE,
    )

    accumulator_int32 = conv2d_int32_reference(
        input_uint8,
        weights_int8,
        bias_int32,
        padding=1,
    )

    actual_float = dequantize_conv_output(
        accumulator_int32,
        weight_scales,
    )

    absolute_error = (
        expected_float - actual_float
    ).abs()

    assert accumulator_int32.dtype == torch.int32
    assert actual_float.shape == expected_float.shape
    assert absolute_error.mean().item() < 0.01
    assert absolute_error.max().item() < 0.03

def test_relu_max_pool2d_int32() -> None:
    accumulator = torch.tensor(
        [[[
            [-4, 2, 3, 1],
            [5, 4, -2, 7],
            [1, 9, 6, 2],
            [3, 0, 8, -1],
        ]]],
        dtype=torch.int32,
    )

    result = relu_max_pool2d_int32(accumulator)

    expected = torch.tensor(
        [[[
            [5, 7],
            [9, 8],
        ]]],
        dtype=torch.int32,
    )

    assert result.dtype == torch.int32
    assert torch.equal(result, expected)


def test_accumulator_requantization() -> None:
    accumulator = torch.tensor(
        [[[
            [-100, 0],
            [1000, 120000],
        ]]],
        dtype=torch.int32,
    )

    weight_scales = torch.tensor(
        [0.0025],
        dtype=torch.float32,
    )

    output_uint8 = requantize_accumulator_uint8(
        accumulator,
        weight_scales,
        output_scale=CONV1_OUTPUT_SCALE,
    )

    assert output_uint8.dtype == torch.uint8
    assert output_uint8.min().item() >= 0
    assert output_uint8.max().item() <= 255

    # Negative accumulators are removed by the implicit ReLU.
    assert output_uint8[0, 0, 0, 0].item() == 0

    # Zero must remain exactly zero.
    assert output_uint8[0, 0, 0, 1].item() == 0

    # A sufficiently large positive value must saturate at 255.
    assert output_uint8[0, 0, 1, 1].item() == 255


def test_uint8_activation_round_trip() -> None:
    activation = torch.tensor(
        [0, 64, 128, 255],
        dtype=torch.uint8,
    )

    reconstructed = dequantize_activation_uint8(
        activation,
        CONV1_OUTPUT_SCALE,
    )

    assert reconstructed.dtype == torch.float32
    assert reconstructed[0].item() == 0.0
    assert torch.isclose(
        reconstructed[-1],
        torch.tensor(
            255.0 * CONV1_OUTPUT_SCALE,
            dtype=torch.float32,
        ),
    )

def test_fixed_requantization_rounding_and_saturation() -> None:
    shift = 20

    # This represents a real multiplier of exactly 0.5.
    integer_multipliers = torch.tensor(
        [1 << (shift - 1)],
        dtype=torch.int32,
    )

    accumulator = torch.tensor(
        [[[
            [-10, 0, 1],
            [2, 510, 600],
        ]]],
        dtype=torch.int32,
    )

    output = requantize_accumulator_uint8_fixed(
        accumulator,
        integer_multipliers,
        shift=shift,
    )

    expected = torch.tensor(
        [[[
            [0, 0, 1],
            [1, 255, 255],
        ]]],
        dtype=torch.uint8,
    )

    assert torch.equal(output, expected)


def test_fixed_requantization_matches_float_reference() -> None:
    torch.manual_seed(42)

    accumulator = torch.randint(
        low=-200000,
        high=200001,
        size=(2, 16, 8, 8),
        dtype=torch.int32,
    )

    weight_scales = torch.linspace(
        0.0020,
        0.0028,
        steps=16,
        dtype=torch.float32,
    )

    float_output = requantize_accumulator_uint8(
        accumulator,
        weight_scales,
        output_scale=CONV1_OUTPUT_SCALE,
    )

    integer_multipliers = (
        calculate_requantization_multipliers(
            weight_scales,
            output_scale=CONV1_OUTPUT_SCALE,
            shift=CONV1_REQUANTIZATION_SHIFT,
        )
    )

    fixed_output = requantize_accumulator_uint8_fixed(
        accumulator,
        integer_multipliers,
        shift=CONV1_REQUANTIZATION_SHIFT,
    )

    difference = (
        float_output.to(torch.int16)
        - fixed_output.to(torch.int16)
    ).abs()

    assert integer_multipliers.dtype == torch.int32
    assert integer_multipliers.min().item() > 0

    # Fixed-point multiplier approximation may differ by one UINT8 level.
    assert difference.max().item() <= 1