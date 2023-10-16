from optimizers.mixop.base_mixop import MixOp
import torch
import torch.nn.functional as F
from optimizers.mixop.entangle import EntangleMixOp


class DiscretizeMixOpV2(EntangleMixOp):

    def preprocess_weights(self, weights):
        return weights
    
    def preprocess_combi(self, weights):

        out = 0
        if len(weights) == 2:
            weights_argmaxed_0 = weights[0]
            weights_argmaxed_1 = weights[1]
            out = weights_argmaxed_0.reshape(weights_argmaxed_0.shape[0], 1) @ weights_argmaxed_1.reshape(1, weights_argmaxed_1.shape[0])
            out = out.flatten()
        elif len(weights) == 3:
            out = weights[0].reshape(weights[0].shape[0], 1) @ weights[1].reshape(1, weights[1].shape[0])
            out = out.flatten()
            out = out.reshape(out.shape[0], 1) @ weights[2].reshape(1, weights[2].shape[0])
            out = out.flatten()
        return out
        
    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        out = 0
        weights = self.preprocess_weights(weights)
        for w, x in zip(weights, x_list):
            out = out + w * x
        params = 0
        if add_params == True:
            for w, param in zip(weights, params_list):
                params = params + w * param
            return out, params
        else:
            return out
