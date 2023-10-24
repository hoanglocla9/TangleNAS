from optimizers.mixop.base_mixop import MixOp
import torch


from optimizers.mixop.entangle import EntangleMixOp, EntangledOp

class SPOSMixOp(MixOp):

    def preprocess_weights(self, weights):
        return weights

    def preprocess_combi(self, weights1, weights2):
        weights = weights1.reshape(weights1.shape[0], 1) @ weights2.reshape(
            1, weights2.shape[0])
        return weights.flatten()

    def transform_weights(self, weights, merge_indices):
        weights_new = []
        for i in range(weights.shape[-1]):
            weights_new.append(weights[i])
        for x in merge_indices:
            weights_new[x[0]]= 0
        i = 0
        for x in merge_indices:
            del weights_new[x[1]-i]
            i = i+1
        weights_new = torch.Tensor(weights_new).cuda()
        return weights_new

    def forward(self, x, weights, ops, add_params=False, combi=False,  merge_indices=None):
        out = 0
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])
        params = 0
        weights = weights.long()
        selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        if merge_indices!=None:
            i = 0
            for m in merge_indices:
                if selected_index == m[0]:
                   return ops[selected_index-i]([x,weights[selected_index:selected_index+2]])
                elif selected_index == m[1]:
                   return ops[selected_index-i-1]([x, weights[(selected_index-1):(selected_index+1)]])
                i = i+1
            weights = self.transform_weights(weights,merge_indices)
            selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        out = ops[selected_index](x)
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
        weights = weights.long()
        selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        #print(selected_index)
        out = ops[selected_index](x, master_op)
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
        weights = weights.long()
        selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        out1, out2 = ops[selected_index](x, master_op)
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
        weights = weights.long()
        selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        out = ops[selected_index](x1, x2, master_op)
        return out

    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        out = 0
        weights = self.preprocess_weights(weights)
        weights = weights.long()
        selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        out = x_list[selected_index]
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
        #print(weights[0])
        #print(weights[1])
        if not combi:
            weights = self.preprocess_weights(weights)
        else:
            weights = self.preprocess_combi(weights[0], weights[1])

        params = 0
        weights = weights.long()
        selected_index = (weights == 1).nonzero(as_tuple=True)[0]
        #print(selected_index)
        out = ops[selected_index](x, mask, B_, N)

        if add_params == True:
            for w, op in zip(weights, ops):
                params = params + w * op.get_parameters()
            return out, params
        else:
            return out
        

class SPOSMixOpV2(EntangleMixOp):

    def preprocess_weights(self, weights):
        return weights

    def preprocess_combi(self, weights):
        out = 0
        if len(weights) == 2:
            out = weights[0].reshape(weights[0].shape[0], 1) @ weights[1].reshape(1, weights[1].shape[0])
            out = out.flatten()
        elif len(weights) == 3:
            out = weights[0].reshape(weights[0].shape[0], 1) @ weights[1].reshape(1, weights[1].shape[0])
            out = out.flatten()
            out = out.reshape(out.shape[0], 1) @ weights[2].reshape(1, weights[2].shape[0])
            out = out.flatten()
        return out

    def forward_depth(self, x_list, weights, params_list=[], add_params=False):
        i = torch.argmax(weights)
        out = weights[i] * x_list[i]
        params = 0
        if add_params == True:
            for w, param in zip(weights, params_list):
                params = params + w * param
            return out, params
        else:
            return out

    def forward(self, x, weights, ops, add_params=False, combi=False):
        """ Forward pass through the MixedOp

        add_params and combi are ignored and do not have any effect
        """
        if combi == True:
            weights = self.preprocess_combi(weights)  
        else:
            weights = self.preprocess_weights(weights)
        argmax = torch.argmax(weights).item()

        chosen_op = ops[argmax]
        
        if isinstance(chosen_op, EntangledOp):
            # Find out the weight of the other EntangledOp
            # Then call forward on entangled_op with the non-None op with use_argmax=True
            entangled_op_weights = []
            entangled_op = None
            i = 0
            for w, op in zip(weights, ops):
                if isinstance(op, EntangledOp):
                    if chosen_op.name == op.name:
                        entangled_op_weights.append(w)
                        if op.op is not None:
                            entangled_op = op
                i = i+1

            #assert len(entangled_op_weights) == 2 # Assuming only two operations are entangled at once
            return entangled_op(x,entangled_op_weights, use_argmax=True)
        else:
            return weights[argmax] * chosen_op(x)


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
            weights = self.preprocess_combi(weights)
        argmax = torch.argmax(weights).item()

        chosen_op = ops[argmax]
        if isinstance(chosen_op, EntangledOp):
            # Find out the weight of the other EntangledOp
            # Then call forward on entangled_op with the non-None op with use_argmax=True
            entangled_op_weights = []
            entangled_op = None

            for w, op in zip(weights, ops):
                if isinstance(op, EntangledOp):
                    if chosen_op.name == op.name:
                        entangled_op_weights.append(w)
                        if op.op is not None:
                            entangled_op = op

            return entangled_op(x, entangled_op_weights, use_argmax=True)
        else:
            return weights[argmax] * chosen_op(x, master_op)