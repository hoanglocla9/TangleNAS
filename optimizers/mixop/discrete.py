from optimizers.mixop.base_mixop import MixOp
import torch


class DiscretizeMixOp(MixOp):

    def __init__(self):
        super(DiscretizeMixOp, self).__init__()

    def preprocess_weights(self, weights):
        weights = torch.nn.functional.softmax(weights, dim=-1)
        return weights

    def preprocess_combi(self, weights1, weights2):
        x1 = torch.softmax(weights1, dim=-1)
        x2 = torch.softmax(weights2, dim=-1)
        weights = x1.reshape(x1.shape[0], 1) @ x2.reshape(1, x2.shape[0])
        return weights.flatten()

    def forward(self, x, weights, ops, add_params=False, combi=False):
        index = weights.argmax().item()
        return ops[index](x)

    def forward_progressive(self,
                            x,
                            weights,
                            ops,
                            add_params=False,
                            combi=False):
        index = weights.argmax().item()
        return ops[index](x)

    def forward_layer(self,
                      x,
                      weights,
                      ops,
                      master_op,
                      add_params=False,
                      combi=False):
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])
        index = weights.argmax().item()
        return ops[index](x, master_op)

    def forward_layer_2_outputs(self,
                                x,
                                weights,
                                ops,
                                base_op,
                                add_params=False):
        pass

    def forward_layer_2_inputs(self,
                               x1,
                               x2,
                               weights,
                               ops,
                               base_op,
                               add_params=False):
        pass

    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        pass

    def forward_swin_attn(self,
                          x,
                          weights,
                          ops,
                          mask,
                          B_,
                          N,
                          add_params=False,
                          combi=False):
        pass
