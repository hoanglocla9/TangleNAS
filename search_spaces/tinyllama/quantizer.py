import torch 
import torch.distributed as dist
import random
import torch.nn as nn
from quant_utils import _choose_qparams_per_token_asymmetric, _fake_quantize_per_token, _fake_quantize_per_channel_group
from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)
# code from https://github.com/zhutmost/lsq-net

# def grad_scale(x, scale):
#     y = x
#     y_grad = x * scale
#     return (y-y_grad).detach()+y_grad


# def round_pass(x):
#     y = x.round()
#     y_grad = x
#     return (y-y_grad).detach()+y_grad


# def quantize(x, s, thd_neg, thd_pos, scale_grad=False):
#     if scale_grad:
#         s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)
#         s_scale = grad_scale(s, s_grad_scale)
#     else:
#         s_scale = s
    
#     x = x / s_scale
#     x = x.clamp(min=thd_neg, max=thd_pos)
#     x = round_pass(x) 
#     x = x * s_scale
#     return x




class Quantizer(nn.Module):
    def __init__(self, abit_list=[16], wbit_list=[16], scale_grad=True):
        super().__init__()
        self.abit_list = abit_list
        self.wbit_list = wbit_list
        
        # self.a_scale = torch.nn.Parameter(torch.ones(len(self.abit_list)))
        self.w_scales = [None for i in range(len(self.wbit_list))]
        self.a_scales = [None for i in range(len(self.abit_list))]
        self.w_zero_points = [None for i in range(len(self.wbit_list))]
        self.a_zero_points = [None for i in range(len(self.abit_list))]
        self.skip_init = False
        self.abit_mapping = {}
        for bit_idx, bit in enumerate(self.abit_list):
            self.abit_mapping[bit] = bit_idx
        self.wbit_mapping = {}
        for bit_idx, bit in enumerate(self.wbit_list):
            self.wbit_mapping[bit] = bit_idx
        self.scale_gradient = scale_grad
        

    def __repr__(self):
        return f'Quantizer. Activation Bit-width candidates: {self.abit_list}, Weight Bit-width candidates: {self.wbit_list}, gradient scaling: {self.scale_gradient}'

    def _index_abits(self, bit):
        if isinstance(bit, torch.Tensor):
            bit = bit.cpu().item()
        return self.abit_mapping[bit]
    
    def _index_wbits(self, bit):
        if isinstance(bit, torch.Tensor):
            bit = bit.cpu().item()
        return self.wbit_mapping[bit]
    
    def init_qparams_for_weights(self, org_weight):
        for wbits in self.wbit_list:
            if wbits <= 8:
                self._per_channel_forward(org_weight, wbits)
    # def sample(self, cands, max=False, min=False):
    #     if max:
    #         bit_width = cands.max()
    #     elif min:
    #         bit_width = cands.min()
    #     else:
    #         bit_width = random.choice(cands)

    #     return bit_width.cpu().item()
        
    # def get_ascale(self, bit_width, detach=True):
    #     s = self.a_scale[self._index_abits(bit_width)]
    #     if detach:
    #         s = s.detach()
    #     return s
    
    # def get_wscale(self, bit_width, detach=True):
    #     s = self.w_scale[self._index_wbits(bit_width)]
    #     if detach:
    #         s = s.detach()
    #     return s
    
    def compute_thd(self, bits, symmetric=True, all_positive=False):
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            thd_neg = 0
            thd_pos = 2 ** bits - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                thd_neg = - 2 ** (bits - 1) # + 1
                thd_pos = 2 ** (bits - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                thd_neg = - 2 ** (bits - 1)
                thd_pos = 2 ** (bits - 1) - 1
        
        if isinstance(thd_neg, torch.Tensor):
            thd_neg = int(thd_neg.cpu().item())
            thd_pos = int(thd_pos.cpu().item())
        elif isinstance(thd_neg, float):
            thd_neg = int(thd_neg)
            thd_pos = int(thd_pos)
    
        return thd_neg, thd_pos

    def _per_token_forward(self, x: torch.Tensor, bit_width:int):
        """
        Perform per token fake quantization on the tensor. 
        For activation
        """
        
        abit_idx = self._index_abits(bit_width)
        (a_scale, zero_point) = _choose_qparams_per_token_asymmetric(
                    x,
                    torch.float32,
                    torch.float32,
                )
        self.a_scales[abit_idx] = a_scale.to(x.device)
        self.a_zero_points[abit_idx] = zero_point.to(x.device)

        # if self.a_scales[abit_idx].device != x.device:
        #     self.a_scales[abit_idx] = self.a_scales[abit_idx].to(x.device)
        #     self.a_zero_points[abit_idx] = self.a_zero_points[abit_idx].to(x.device)

        qmin, qmax = self.compute_thd(bit_width, symmetric=False, all_positive=True)
        return _fake_quantize_per_token(x, self.a_scales[abit_idx], self.a_zero_points[abit_idx], qmin, qmax)

    def _per_channel_forward(self, x: torch.Tensor, bit_width:int):
        """
        Perform per channel or per group fake quantization on the tensor.
        We express per channel using per group where the group size is the size
        of the last dimension of the tensor.
        For weight
        """
        # get group size
        ## support channel quantization
        group_size = x.size()[-1]

        # get scales and zero points

        wbit_idx = self._index_wbits(bit_width)
        if self.w_scales[wbit_idx] is None or self.w_zero_points[wbit_idx] is None:
            (scale, zero_point) = get_group_qparams_symmetric(
                    x,
                    bit_width,
                    group_size,
                    torch.float32,
            )
            zero_point = zero_point.to(torch.float32)
            self.w_scales[wbit_idx] = scale.to(x.device)
            self.w_zero_points[wbit_idx] = zero_point.to(x.device)

        if self.w_scales[wbit_idx].device != x.device:
            self.w_scales[wbit_idx] = self.w_scales[wbit_idx].to(x.device)
            self.w_zero_points[wbit_idx] = self.w_zero_points[wbit_idx].to(x.device)

        qmin, qmax =  self.compute_thd(bit_width, symmetric=True, all_positive=False)
        return _fake_quantize_per_channel_group(
            x,
            self.w_scales[wbit_idx],
            self.w_zero_points[wbit_idx],
            qmin,
            qmax,
            group_size,
            # zero_point_domain,
        )
    
    def forward(self, x, weights, abits, wbits):
        ### Currently only support for 1 precision of a
        assert x is not None or weights is not None, 'activation or weight need to be not None'
        q_x, q_weights = x, weights

        if x is not None and abits <= 8:
            q_x = self._per_token_forward(x, abits)

        if weights is not None and wbits <= 8:
            q_weights = self._per_channel_forward(weights, wbits)

        return q_x, q_weights
    
    # def weight_bound(self, bits, scale=None):
    #     lower, upper = self.compute_thd(bits)
    #     if scale is None:
    #         step_size = self.get_wscale(bits)
    #     else:
    #         step_size = scale
    #     return step_size * lower, step_size * upper