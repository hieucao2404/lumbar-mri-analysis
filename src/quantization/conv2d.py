import torch
from torch.nn import functional as F

UINT8_SCALE = 1.0 / 255.0

def quantize_input_uint8(
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Convert an input tensor from [0.0, 1.0] to UINT8 [0, 255]
    The zero point is 0 and the scale is 1 / 255
    """
    return (
        torch.round(input_tensor.clamp(0.0, 1.0) * 255.0)
        .clamp(0, 255)
        .to(torch.uint8)
    )

def quantize_weights_int8_per_channel(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize convolution weights to signed INT8.
    
    A separate scale is calculated for every output channel.

    Input shape:
        [output_channels, input_channels, kernel_height, kernel_width]

    Returns:
        quantized_weights: INT8 tensor with the same shape
        weight_scales: FLOAT32 tensor shaped [output_channels]
    """
    maximum_absolute_values = (
        weights.abs()
        .flatten(start_dim=1)
        .amax(dim=1)
    )

    weight_scales = maximum_absolute_values / 127.0

    #Prevent division by zero for an entirely zero-valued channel.
    weight_scales = torch.where(
        weight_scales > 0,
        weight_scales,
        torch.ones_like(weight_scales),
    )

    quantized_weights = torch.round(
        weights / weight_scales[:, None, None, None]
    )

    quantized_weights = quantized_weights.clamp(
        -127,
        127,
    ).to(torch.int8)

    return quantized_weights, weight_scales

def quantize_bias_int32(
    bias: torch.Tensor,
    weight_scales: torch.Tensor,
    input_scale: float = UINT8_SCALE,
) -> torch.Tensor:
    """
    Quantize each bias using:
        bias_scale[channel] =
            input_scale * weight_scale[channel]
    """
    bias_scales = input_scale * weight_scales

    return torch.round(
        bias/bias_scales
    ).to(torch.int32)


def conv2d_int32_reference(
    input_uint8: torch.Tensor,
    weights_int8: torch.Tensor,
    bias_int32: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
) -> torch.Tensor:
    """
    Perform convolution using integer arithmetic.

    Inputs: 
        input_uint8: [N. C_in, H, W]
        weights_int8: [C_out, C_in, K_h, K_w]
        bias_int32: [C_out]

    Output:
        Signed INT32 accumulator tensor:
        [N, C_out, H_out, W_out]
    
    This implementation is intended as a bit-accurate reference for RTL,
    not as a high-performance convolution implementation
    """
    if input_uint8.dtype != torch.uint8:
        raise TypeError("input_uint8 must have dtype torch.uint8")

    if weights_int8.dtype != torch.int8:
        raise TypeError("weights_int8 must have dtype torch.int8")

    if bias_int32.dtype != torch.int32:
        raise TypeError("bias_int32 must have dtype torch.int32")

    if input_uint8.device.type != "cpu":
        raise ValueError(
            "The integer reference convolution must run on the CPU."
        )

    batch_size, input_channels, _, _ = input_uint8.shape

    output_channels, weight_input_channels, kernel_height, kernel_width =(
        weights_int8.shape
    )

    if input_channels != weight_input_channels:
        raise ValueError(
            "Input channels do not match the weight channels."
        )

    padded_input = F.pad(
        input_uint8.to(torch.int32),
        (padding, padding, padding, padding),
        mode="constant",
        value=0,
    )

    patches = (
        padded_input
        .unfold(2, kernel_height, stride)
        .unfold(3, kernel_width, stride)
    )

    output_height = patches.shape[2]
    output_width = patches.shape[3]

    patches = (
        patches
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(
            batch_size,
            output_height,
            output_width,
            input_channels * kernel_height * kernel_width,
        )
    )

    flattened_weights = (
        weights_int8
        .to(torch.int32)
        .view(output_channels, -1)
    )

    accumulator = torch.matmul(
        patches,
        flattened_weights.t(),
    )

    accumulator = accumulator + bias_int32.view(
        1,
        1,
        1,
        output_channels,
    )

    return (
        accumulator
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(torch.int32)
    )

def dequantize_conv_output(
    accumulator_int32: torch.Tensor,
    weight_scales: torch.Tensor,
    input_scale: float = UINT8_SCALE,
) -> torch.Tensor:
    """
    Convert the INT32 accumulator back to floating point.
    """
    output_scales = (
        input_scale * weight_scales
    ).view(1, -1, 1, 1)

    return accumulator_int32.to(torch.float32) * output_scales

def relu_max_pool2d_int32(
        accumulator_int32: torch.Tensor,
        kernel_size: int = 2,
        stride: int = 2,
) -> torch.Tensor:
    """
    Appky ReLU and max pooling directly to an INT32 accumulator.

    Keeping the values as INT32 until after pooling reduces the
    number of requantization operations required by the hardware
    """
    if accumulator_int32.dtype != torch.int32:
        raise TypeError(
            "accumulator_int32 must have dtype torch.int32"
        )

    relu_output = accumulator_int32.clamp_min(0)

    patches = (
        relu_output
        .unfold(2, kernel_size, stride)
        .unfold(3, kernel_size, stride)
    )

    return patches.amax(dim=(-1, -2))

def requantize_accumulator_uint8(
    accumulator_int32: torch.Tensor,
    weight_scales: torch.Tensor,
    output_scale: float,
    input_scale: float = UINT8_SCALE,
) -> torch.Tensor:
    """
    Convert an INT32 accumulator into a shared-scale UINT8 output.

    For each output channel:
        real_multiplier =
            input_scale * weight_scale/ output_scale
        
            output_uint8 = clamp(round(accumulator * real_multiplier), 0, 255)
    
    The lower clamp also performs ReLU.
    """
    if accumulator_int32.dtype != torch.int32:
        raise TypeError(
            "accumulator_int32 must have dtype torch.int32"
        )

    if output_scale <= 0:
        raise ValueError("output_scale must be positive")

    real_multipliers = (
        input_scale * weight_scales / output_scale
    ).view(1, -1, 1, 1)

    scaled_output = (
        accumulator_int32.to(torch.float32) * real_multipliers
    )

    return (
        torch.round(scaled_output)
        .clamp(0, 255)
        .to(torch.uint8)
    )

def dequantize_activation_uint8(
    activation_uint8: torch.Tensor,
    output_scale: float,
) -> torch.Tensor:
    """
    Convert a zero-point-zero UINT8 activation back to float
    """
    if activation_uint8.dtype != torch.uint8:
        raise TypeError(
            "activation_uint8 muist have dtype torch.uint8"
        )

    return (
        activation_uint8.to(torch.float32)
        * output_scale
    )


def calculate_requantization_multipliers(
    weight_scales: torch.Tensor,
    output_scale: float,
    input_scale: float = UINT8_SCALE,
    shift: int = 20,
) -> torch.Tensor:
    """
    Convert floating-point channel multipliers into fixed-point integers.

    real_multiplier[channel] =
        input_scale * weight_scale[channel] / output_scale

    integer_multiplier[channel] = 
        round(real_multiplier[channel] * 2**shift)
    """
    if output_scale <= 0:
        raise ValueError("output_scale must be positive")

    if shift <= 0:
        raise ValueError("shift must be positive")

    real_multipliers = (
        input_scale * weight_scales / output_scale
    )

    integer_multipliers = torch.round(
        real_multipliers * (1 << shift)
    )

    return integer_multipliers.to(torch.int32)

def requantize_accumulator_uint8_fixed(
        accumulator_int32: torch.Tensor,
        integer_multipliers: torch.Tensor,
        shift: int = 20,
) -> torch.Tensor:
    """
    Requantize an INT32 accumulator using integer-only arithmetic

    For each channel:
        positive_accumulator = max(accumulator, 0)

        rounded_product = 
            positive_accumulator * integer_multiplier
            + 2 **(shift - 1)

        output = 
            clamp(rounded_product >> shift, 0, 255)

    The addition before shifting implements round-half-up behavior 
    """
    if accumulator_int32.dtype != torch.int32:
        raise TypeError(
            "accumulator_int32 must have dtype torch.int32"
        )

    if integer_multipliers.dtype != torch.int32:
        raise TypeError(
            "integer_multipliers must have dtype torch.int32"
        )

    if shift <= 0:
        raise ValueError("shift must be positive")

    positive_accumulator = (
        accumulator_int32
        .clamp_min(0)
        .to(torch.int64)
    )

    multipliers = integer_multipliers.to(
        torch.int64
    ).view(1, -1, 1, 1)

    product = positive_accumulator * multipliers

    rounding_offset = 1 << (shift - 1)

    rounded_output = (
        product + rounding_offset
    ) >> shift

    return (
        rounded_output
        .clamp(0, 255)
        .to(torch.uint8)
    )