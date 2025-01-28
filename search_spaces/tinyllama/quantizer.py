import torch 
import torch.distributed as dist
import random
import torch.nn as nn



# code from https://github.com/zhutmost/lsq-net

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y-y_grad).detach()+y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y-y_grad).detach()+y_grad


def quantize(x, s, thd_neg, thd_pos, scale_grad=False):
    if scale_grad:
        s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(s, s_grad_scale)
    else:
        s_scale = s
    
    x = x / s_scale
    x = x.clamp(min=thd_neg, max=thd_pos)
    x = round_pass(x) 
    x = x * s_scale
    return x




class Quantizer(nn.Module):
    def __init__(self, abit_list=[16], wbit_list=[16], scale_grad=True):
        super().__init__()
        self.abit_list = abit_list
        self.wbit_list = wbit_list
        
        # self.a_scale = torch.nn.Parameter(torch.ones(len(self.abit_list)))
        self.w_scale = torch.nn.Parameter(torch.ones(len(self.wbit_list)))
        self.skip_init = False
        self.abit_mapping = {}
        for bit_idx, bit in enumerate(self.abit_list):
            self.abit_mapping[bit] = bit_idx
        self.wbit_mapping = {}
        for bit_idx, bit in enumerate(self.wbit_list):
            self.wbit_mapping[bit] = bit_idx
        self.register_buffer('a_init_state', torch.zeros(len(abit_list)))
        self.scale_gradient = scale_grad
        
    def init_weight_scale(self, previous_layer_weights):
        self.w_scale = torch.nn.Parameter(torch.ones(len(self.wbit_list)))
        mean = previous_layer_weights.detach().mean() 
        std = previous_layer_weights.detach().std() 
        for i, b in enumerate(self.wbit_list): 
            s_init = torch.max((mean-3*std).abs(), (mean+3*std).abs())/2**(max(self.wbit_list)-1) * 2 ** (max(self.wbit_list) - b)
            self.w_scale[i].data.copy_(s_init)

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
    
    def sample(self, cands, max=False, min=False):
        if max:
            bit_width = cands.max()
        elif min:
            bit_width = cands.min()
        else:
            bit_width = random.choice(cands)

        return bit_width.cpu().item()
        
    def get_ascale(self, bit_width, detach=True):
        s = self.a_scale[self._index_abits(bit_width)]
        if detach:
            s = s.detach()
        return s
    
    def get_wscale(self, bit_width, detach=True):
        s = self.w_scale[self._index_wbits(bit_width)]
        if detach:
            s = s.detach()
        return s
    
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

    def forward(self, x, weights, abits, wbits, skip_init=False, scale_a=None, scale_w=None, **args):
        ### Currently only support for 1 precision of a
        q_x, q_weights = None, None
        assert x is not None or weights is not None, 'activation or weight need to be not None'

        if x is not None:
            # idx = self._index_abits(abits)
            # thd_neg, thd_pos = self.compute_thd(abits, symmetric=True, all_positive=False)
            # if self.a_init_state[idx] == 0 and not skip_init:
            #     self.a_init_state[idx].fill_(1)
            #     s_init = x.detach().abs().mean() * 2 / (thd_pos ** 0.5)
            #     self.a_scale[idx].data.copy_(s_init)
            # if scale_a is None:
            #     a_s = self.a_scale[idx]
            # else:
            #     a_s = scale_a
            # q_x = quantize(x, a_s, thd_neg, thd_pos, scale_grad=self.scale_gradient)
            q_x = x
        if weights is not None:
            idx = self._index_wbits(wbits)
            thd_neg, thd_pos = self.compute_thd(wbits, symmetric=True, all_positive=False)

            if scale_w is None:
                w_s = self.w_scale[idx]
            else:
                w_s = scale_w
                
            q_weights = quantize(weights, w_s, thd_neg, thd_pos, scale_grad=self.scale_gradient)

        return q_x, q_weights
    
    def weight_bound(self, bits, scale=None):
        lower, upper = self.compute_thd(bits)
        if scale is None:
            step_size = self.get_wscale(bits)
        else:
            step_size = scale
        return step_size * lower, step_size * upper