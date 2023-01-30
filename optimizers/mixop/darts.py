from optimizers.mixop.base_mixop import MixOp
import torch

from optimizers.mixop.entangle import EntangleMixOp, EntangledOp

class DARTSMixOp(MixOp):

    def preprocess_weights(self, weights):
        weights = torch.nn.functional.softmax(weights, dim=-1)
        return weights

    def preprocess_combi(self, weights1, weights2):
        x1 = torch.softmax(weights1, dim=-1)
        x2 = torch.softmax(weights2, dim=-1)
        weights = x1.reshape(x1.shape[0], 1) @ x2.reshape(1, x2.shape[0])
        return weights.flatten()

    def forward(self, x, weights, ops, add_params=False, combi=False):
        out = 0
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])
        params = 0
        for w, op in zip(weights, ops):
            out = out + w * op(x)

        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out

    def forward_progressive(self,
                            x,
                            weights,
                            ops,
                            add_params=False,
                            combi=False):
        out = 0
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])
        params = 0
        for w, op in zip(weights, ops):
            if not w == 0:
                out = out + w * op(x)

        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out

    def forward_layer(self,
                      x,
                      weights,
                      ops,
                      master_op,
                      add_params=False,
                      combi=False):
        out = 0
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])
        params = 0
        for w, op in zip(weights, ops):
            out = out + w * op(x, master_op)
        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out

    def forward_layer_2_outputs(self,
                                x,
                                weights,
                                ops,
                                master_op,
                                add_params=False):
        out1 = 0
        out2 = 0
        params = 0
        weights = self.preprocess_weights(weights)
        for w, op in zip(weights, ops):
            x1, x2 = op(x, master_op)
            out1 = out1 + w * x1
            out2 = out2 + w * x2
        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out1, out2, params
        else:
            return out1, out2

    def forward_layer_2_inputs(self,
                               x1,
                               x2,
                               weights,
                               ops,
                               master_op,
                               add_params=False):
        out = 0
        weights = self.preprocess_weights(weights)
        for w, op in zip(weights, ops):
            out = out + w * op(x1, x2, master_op)
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

    def forward_swin_attn(self,
                          x,
                          weights,
                          ops,
                          mask,
                          B_,
                          N,
                          add_params=False,
                          combi=False):
        out = 0
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])
        params = 0
        for w, op in zip(weights, ops):
            out = out + w * op(x, mask, B_, N)

        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out


class DARTSMixOpV2(EntangleMixOp):

    def preprocess_weights(self, weights):
        weights = torch.nn.functional.softmax(weights, dim=-1)
        return weights
