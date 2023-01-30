#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################

import torch
import torch.nn as nn
from copy import deepcopy

from search_spaces.NATS.operations import OPS
from search_spaces.NATS.operations import OPS_sub
from optimizers.optim_factory import get_mixop
import itertools
# Cell for NAS-Bench-201


class InferCellV1(nn.Module):

    def __init__(self,
                 mixop,
                 genotype,
                 C_in,
                 C_out,
                 stride,
                 affine=False,
                 track_running_stats=False):
        super(InferCellV1, self).__init__()
        channels_choice = [8, 16, 24, 32, 40, 48, 56, 64]
        self.layers = nn.ModuleList()
        self.ops_list = []
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        self.mixop = mixop
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                #print(op_name)
                if op_in == 0:
                    layer = OPS[op_name](64, 64, stride, affine,
                                         track_running_stats)
                else:
                    layer = OPS[op_name](64, 64, 1, affine,
                                         track_running_stats)
                if op_name == "nor_conv_1x1" or op_name == "nor_conv_3x3":
                    self.ops_list.append([[
                        OPS_sub[op_name + "_p1"](c, c, max(channels_choice), 1)
                        for c in channels_choice
                    ],
                                          [
                                              OPS_sub[op_name + "_p2"](
                                                  c, c, max(channels_choice),
                                                  1) for c in channels_choice
                                          ]])
                else:
                    self.ops_list.append([
                        OPS_sub[op_name](c1, c2, max(channels_choice), 1)
                        for (c1, c2) in itertools.product(
                            channels_choice, channels_choice)
                    ])
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out
        self.relu = torch.nn.ReLU()

    def extra_repr(self):
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__)
        laystr = []
        for i, (node_layers,
                node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [
                "I{:}-L{:}".format(_ii, _il)
                for _il, _ii in zip(node_layers, node_innods)
            ]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (string + ", [{:}]".format(" | ".join(laystr)) +
                ", {:}".format(self.genotype.tostr()))

    def forward(self, inputs, alphas1, alphas2):
        nodes = [inputs]
        #print(inputs.shape)
        assert len(self.layers) == len(self.ops_list)

        for i, (node_layers,
                node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = 0
            for _il, _ii in zip(node_layers, node_innods):
                input = nodes[_ii]
                if self.layers[_il].__class__.__name__ == "ReLUConvBN":
                    node_feature_p1 = self.mixop.forward_layer(
                        input, alphas1, self.ops_list[_il][0],
                        self.layers[_il])
                    node_feature_p2 = self.mixop.forward_layer(
                        node_feature_p1, alphas2, self.ops_list[_il][1],
                        self.layers[_il])
                    node_feature = node_feature + node_feature_p2
                else:
                    node_feature = node_feature + self.mixop.forward_layer(
                        input, [alphas1, alphas2],
                        self.ops_list[_il],
                        self.layers[_il],
                        combi=True)
            nodes.append(node_feature)
        return nodes[-1]


class InferCellV2(nn.Module):

    def __init__(self,
                 mixop,
                 genotype,
                 C_in,
                 C_out,
                 stride,
                 affine=True,
                 track_running_stats=True):
        super(InferCellV2, self).__init__()
        channels_choice = [8, 16, 24, 32, 40, 48, 56, 64]
        self.layers = nn.ModuleList()
        self.ops_list = []
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        self.mixop = mixop
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                #print(op_name)
                if op_in == 0:
                    layer = OPS[op_name](64, 64, stride, affine,
                                         track_running_stats)
                else:
                    layer = OPS[op_name](64, 64, 1, affine,
                                         track_running_stats)
                    stride = 1
                self.ops_list.append([
                    OPS_sub[op_name](c1,
                                     c2,
                                     max(channels_choice),
                                     stride=stride) for (c1, c2) in
                    itertools.product(channels_choice, channels_choice)
                ])
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out
        self.relu = torch.nn.ReLU()

    def extra_repr(self):
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__)
        laystr = []
        for i, (node_layers,
                node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [
                "I{:}-L{:}".format(_ii, _il)
                for _il, _ii in zip(node_layers, node_innods)
            ]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (string + ", [{:}]".format(" | ".join(laystr)) +
                ", {:}".format(self.genotype.tostr()))

    def forward(self, inputs, alphas1, alphas2):
        nodes = [inputs]
        #print(inputs.shape)
        assert len(self.layers) == len(self.ops_list)

        for i, (node_layers,
                node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = 0
            #print(len(node_layers))
            #print(len(node_innods))
            for _il, _ii in zip(node_layers, node_innods):
                input = nodes[_ii]
                #print(input.shape)
                node_feature = node_feature + self.mixop.forward_layer(
                    input, [alphas1, alphas2],
                    self.ops_list[_il],
                    self.layers[_il],
                    combi=True)
            nodes.append(node_feature)
            #for i in range(len(nodes)):
            #    print(nodes[i].shape)
        return nodes[-1]


class InferCellDiscretize(nn.Module):

    def __init__(self,
                 genotype,
                 C_in,
                 C_out,
                 stride,
                 affine=True,
                 track_running_stats=True):
        super(InferCellDiscretize, self).__init__()
        channels_choice = [8, 16, 24, 32, 40, 48, 56, 64]
        self.layers = nn.ModuleList()
        self.ops_list = []
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        self.mixop = get_mixop("discrete")
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                #print(op_name)
                if op_in == 0:
                    layer = OPS[op_name](64, 64, stride, affine,
                                         track_running_stats)
                else:
                    layer = OPS[op_name](64, 64, 1, affine,
                                         track_running_stats)
                    stride = 1
                self.ops_list.append([
                    OPS_sub[op_name](c1,
                                     c2,
                                     max(channels_choice),
                                     stride=stride) for (c1, c2) in
                    itertools.product(channels_choice, channels_choice)
                ])
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out
        self.relu = torch.nn.ReLU()

    def extra_repr(self):
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__)
        laystr = []
        for i, (node_layers,
                node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [
                "I{:}-L{:}".format(_ii, _il)
                for _il, _ii in zip(node_layers, node_innods)
            ]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (string + ", [{:}]".format(" | ".join(laystr)) +
                ", {:}".format(self.genotype.tostr()))

    def forward(self, inputs, alphas1, alphas2):
        nodes = [inputs]
        #print(inputs.shape)
        assert len(self.layers) == len(self.ops_list)

        for i, (node_layers,
                node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = 0
            for _il, _ii in zip(node_layers, node_innods):
                input = nodes[_ii]
                #print(input.shape)
                node_feature = node_feature + self.mixop.forward_layer(
                    input, [alphas1, alphas2],
                    self.ops_list[_il],
                    self.layers[_il],
                    combi=True)
            nodes.append(node_feature)
        return nodes[-1]
