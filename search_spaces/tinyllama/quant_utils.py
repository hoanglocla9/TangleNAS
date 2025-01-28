import torch
import math
from typing import Tuple, List
from torchao.quantization.utils import (
    _get_per_token_block_size,
)
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    fake_quantize_affine_cachemask,
)
class _GenericFakeQuantize(torch.autograd.Function):
    """
    Implementation of generic fake quantize with backward STE.

    With the appropriate input tensor shape, this can be used to express
    grouped per channel fake quantize or per token fake quantize.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        block_size: List[int],
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        quant_min: int,
        quant_max: int,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    ) -> torch.Tensor:
        # avoid circular dependencies
        from torchao.quantization.qat.affine_fake_quantized_tensor import (
            AffineFakeQuantizedTensor,
        )

        if isinstance(input, AffineFakeQuantizedTensor):
            _input = input.original_tensor
        else:
            _input = input

        (fq, mask) = fake_quantize_affine_cachemask(
            _input,
            block_size,
            scales,
            zero_points,
            torch.int32,
            quant_min,
            quant_max,
            zero_point_domain,
        )

        ctx.save_for_backward(mask)
        return fq

    @staticmethod
    def backward(ctx, gy):
        (mask,) = ctx.saved_tensors
        return gy * mask, None, None, None, None, None, None
    
def get_minq_maxq(bit, sym):
    if sym:
        maxq = torch.tensor(2**(bit-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bit - 1)
        minq = 0
    
    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale

def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def _choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    scales_precision: torch.dtype = torch.float32,
    zero_points_precision: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       scales_precision (torch.dtype): precision of returned scales
       zero_points_precision (torch.dtype): precision of returned zero points

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val = torch.amin(input, dim=-1, keepdim=True)
    max_val = torch.amax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(scales_precision), zero_point.to(zero_points_precision)

def _fake_quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
) -> torch.Tensor:
    # from torch.ao.quantization.fx._decomposed import _per_token_quant_qparam_dim_check

    # _per_token_quant_qparam_dim_check(input, scales, zero_points)
    flexible_dim = input.size()[1]
    scales = scales[:,:flexible_dim,:]
    zero_points = zero_points[:,:flexible_dim,:]
    block_size = _get_per_token_block_size(input)
    fq_input = input.to(torch.float32)
    fq = _GenericFakeQuantize.apply(
        fq_input,
        block_size,
        scales,
        zero_points,
        quant_min,
        quant_max,
    )
    return fq.reshape_as(input).to(input.dtype)

def _fake_quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    group_size: int,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    assert group_size > 1
    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    block_size = (1, group_size)
    flexible_out_dim = input.size()[0]
    scales = scales[:flexible_out_dim,:]
    zero_points = zero_points[:flexible_out_dim,:]
    return _GenericFakeQuantize.apply(
        input,
        block_size,
        scales,
        zero_points,
        quant_min,
        quant_max,
        zero_point_domain,
    )