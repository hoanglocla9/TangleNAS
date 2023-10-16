import collections
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import argparse

import torch

import contextlib
import pickle
from toy_search_spaces.nanoGPT.model_spos import GPT, GPTConfig
# Encoder: take a string, output a list of integers


def encode(s):
    return [stoi[c] for c in s]

# Decoder: take a list of integers, output a string
def decode(l):
    return ''.join([itos[i] for i in l])




def get_batch(train_data, eval_data, test_data, split: str, block_size: int = 8, batch_size: int = 4, device: str = None):
    """ Gets a randomized batch from the split of data chosen.

    Arguments
    ---------
    split : str, {"train", "valid"}
    block_size : int
        The context length for predictions, that is, a sentence length
    batch_size : int
        The batch size, that is, the number of sentences
    """
    # generate a small batch of data of inputs x and targets y
    assert split in ["train", "valid", "test"]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = train_data if split == 'train' else eval_data
    if split == "test":
        data = test_data
    # generating random indices as markers in the full text document
    # such that they are a starting point to the sentence of length
    # `block_size` that will be a data point in the batch
    ix = torch.randint(
        low=0, high=len(data) - block_size, size=(batch_size,)
    )
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix`
    x = torch.stack([data[i:i+block_size] for i in ix])
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix` + 1 (shifted to right)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class NASOptimizer(object):
    """
    Base class for NASBench-101 optimizers. All subclasses should
    inherit from this.
    """

    def __init__(self):
        # get the configuration space
        # configuration (architecture) at each point in time.
        # incumbent_trajectory_error keeps track of the
        # corresponding validation errors of incumbent_trajectory
        self.incumbent_trajectory = []
        self.incumbent_trajectory_error = []
        self.incumbent_trajectory_test_error = []
        self.all_configs_err = {}
        self.curr_wallclock = 0
        self.curr_incumbent = None
        self.curr_incumbent_error = 10000000
        self.eval_iters = 200

    def optimize(self, n_iters: int = 100):
        raise NotImplementedError

    def sample_random_config(self, model):
        """
        Return a randomly sampled configuration.
        """
        # TODO: return one randomly sampled configuration from self.cs
        config, arch_params = model.sample_random_config(sample_max_layers=True)
        return config, arch_params

    @torch.no_grad()
    def estimate_loss(self, config, model):
        out = {}
        model.eval()
        for split in ['valid','test']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = get_batch(self.train_data,self.eval_data,self.test_data,split)
                logits, loss = model(X, Y, config)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out['valid'], out['test']

    def train_and_eval(self, config, model, train_portion=0.5,optimizer="darts_v1"):
        """
        Function that computes the error on the validation split. 
        Since every architecture has already been trained partially, 
        we just to forward props on the pre-trained supernet 
        """

        #print(arch_params)
        n_layer, n_embd, n_head, mlp_ratio = config
        valid_err, test_err = self.estimate_loss(config, model)
        # TODO: check if config is better than current incumbent
        self.all_configs_err[str(config)] = valid_err
        print(str(config))

        # If we find a better validation error, we update the incumbent, else revet to the best current incumbent
        print("Current incumbent error: ", self.curr_incumbent_error)
        print("Valid error: ", valid_err)
        if min(self.curr_incumbent_error, valid_err) == valid_err:
            self.curr_incumbent_error = valid_err
            self.curr_incumbent_test_error = test_err
            self.curr_incumbent = config
            self.incumbent_trajectory.append(config)
            self.incumbent_trajectory_error.append(valid_err)
            self.incumbent_trajectory_test_error.append(test_err)
        else:
            self.incumbent_trajectory.append(self.curr_incumbent)
            self.incumbent_trajectory_error.append(
                self.incumbent_trajectory_error[-1])
            self.incumbent_trajectory_test_error.append(
                self.incumbent_trajectory_test_error[-1])
        print("Current incumbent error: ", self.curr_incumbent_error)
        print("Current incumbent test error: ", self.curr_incumbent_test_error)
        print("Current incumbent: ", self.curr_incumbent)
        incumbent_traj_file = "incumbent_trajectory_rs_"+str(train_portion)+"_"+str(optimizer)+"_"+str(self.exp_name)+".pkl"
        incumbent_traj_error_file = "incumbent_trajectory_error_rs_"+str(train_portion)+"_"+str(optimizer)+"_"+str(self.exp_name)+".pkl"
        incumbent_traj_test_error_file = "incumbent_trajectory_test_error_rs_"+str(train_portion)+"_"+str(optimizer)+"_"+str(self.exp_name)+".pkl"
        with open(incumbent_traj_error_file, "wb") as f:
            pickle.dump(self.incumbent_trajectory_error, f)
        with open(incumbent_traj_file, "wb") as f:
            pickle.dump(self.incumbent_trajectory, f)
        with open(incumbent_traj_test_error_file, "wb") as f:
            pickle.dump(self.incumbent_trajectory_test_error, f)


class RandomSearch(NASOptimizer):
    """
    Algorithm for random search.
    """

    def __init__(self, model_path, model_args, train_data, eval_data, test_data, train_portion=0.5, optimizer="darts_v1", exp_name="random_search"):
        super(RandomSearch, self).__init__()
        self.model_path = model_path
        choices = {}
        self.exp_name = exp_name
        choices["num_layers"] = [2, 4, 6]
        choices["embed_dim"] = [96, 192, 384]
        choices["num_heads"] = [2, 4, 6]
        choices["mlp_ratio"] = [1, 2, 4]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)
        model_state_dict = torch.load(model_path)["model"]
        print(optimizer)
        if optimizer != "spos":
            state_dict = {}
            state_dict = self.match_state_dict(state_dict, model_state_dict)
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        self.model = self.model.to(device)
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.train_portion = train_portion
        self.optimizer = optimizer

    def match_state_dict(self, model, state_dict):
        model["transformer.wte.weight"] = state_dict["token_embedding_table_op.embedding.weight"]
        model["transformer.wpe.weight"] = state_dict["position_embedding_table_op.embedding.weight"]
        model["transformer.h.0.attn.bias"] = state_dict["transformer.h.0.attn.bias"]
        model["transformer.h.1.attn.bias"] = state_dict["transformer.h.0.attn.bias"]
        model["transformer.h.2.attn.bias"] =state_dict["transformer.h.0.attn.bias"]
        model["transformer.h.3.attn.bias"] = state_dict["transformer.h.0.attn.bias"]
        model["transformer.h.4.attn.bias"] = state_dict["transformer.h.0.attn.bias"]
        model["transformer.h.5.attn.bias"] = state_dict["transformer.h.0.attn.bias"]
        model["transformer.ln_f.weight"] = state_dict["ln_f_op.layer_norm.weight"]
        model["lm_head.weight"] = state_dict["lm_head_op.linear_layer.weight"]
        model["transformer.h.0.ln_1.weight"] = state_dict["transformer.h.0.ln1_op.layer_norm.weight"]
        model["transformer.h.0.ln_2.weight"] = state_dict["transformer.h.0.ln2_op.layer_norm.weight"]
        model["transformer.h.0.attn.c_attn.weight"] = state_dict["transformer.h.0.attn.c_attn_op.linear_layer.weight"]
        model["transformer.h.0.attn.c_proj.weight"] = state_dict["transformer.h.0.attn.c_proj_mix_op.linear_layer.weight"]
        model["transformer.h.0.ln_2.weight"] = state_dict["transformer.h.0.ln2_op.layer_norm.weight"]
        model["transformer.h.0.mlp.c_fc.weight"] = state_dict["transformer.h.0.mlp.linear_0_mix_op.linear_layer.weight"]
        model["transformer.h.0.mlp.c_proj.weight"] = state_dict["transformer.h.0.mlp.linear_1_mix_op.linear_layer.weight"]
        model["transformer.h.1.ln_1.weight"] = state_dict["transformer.h.1.ln1_op.layer_norm.weight"]
        model["transformer.h.1.ln_2.weight"] = state_dict["transformer.h.1.ln2_op.layer_norm.weight"]
        model["transformer.h.1.attn.c_attn.weight"] = state_dict["transformer.h.1.attn.c_attn_op.linear_layer.weight"]
        model["transformer.h.1.attn.c_proj.weight"] = state_dict["transformer.h.1.attn.c_proj_mix_op.linear_layer.weight"]
        model["transformer.h.1.ln_2.weight"] = state_dict["transformer.h.1.ln2_op.layer_norm.weight"]
        model["transformer.h.1.mlp.c_fc.weight"] = state_dict["transformer.h.1.mlp.linear_0_mix_op.linear_layer.weight"]
        model["transformer.h.1.mlp.c_proj.weight"] = state_dict["transformer.h.1.mlp.linear_1_mix_op.linear_layer.weight"]
        model["transformer.h.2.ln_1.weight"] = state_dict["transformer.h.2.ln1_op.layer_norm.weight"]
        model["transformer.h.2.ln_2.weight"] = state_dict["transformer.h.2.ln2_op.layer_norm.weight"]
        model["transformer.h.2.attn.c_attn.weight"] = state_dict["transformer.h.2.attn.c_attn_op.linear_layer.weight"]
        model["transformer.h.2.attn.c_proj.weight"] = state_dict["transformer.h.2.attn.c_proj_mix_op.linear_layer.weight"]
        model["transformer.h.2.ln_2.weight"] = state_dict["transformer.h.2.ln2_op.layer_norm.weight"]
        model["transformer.h.2.mlp.c_fc.weight"] = state_dict["transformer.h.2.mlp.linear_0_mix_op.linear_layer.weight"]
        model["transformer.h.2.mlp.c_proj.weight"] = state_dict["transformer.h.2.mlp.linear_1_mix_op.linear_layer.weight"]
        model["transformer.h.3.ln_1.weight"] = state_dict["transformer.h.3.ln1_op.layer_norm.weight"]
        model["transformer.h.3.ln_2.weight"] = state_dict["transformer.h.3.ln2_op.layer_norm.weight"]
        model["transformer.h.3.attn.c_attn.weight"] = state_dict["transformer.h.3.attn.c_attn_op.linear_layer.weight"]
        model["transformer.h.3.attn.c_proj.weight"] = state_dict["transformer.h.3.attn.c_proj_mix_op.linear_layer.weight"]
        model["transformer.h.3.ln_2.weight"] = state_dict["transformer.h.3.ln2_op.layer_norm.weight"]
        model["transformer.h.3.mlp.c_fc.weight"] = state_dict["transformer.h.3.mlp.linear_0_mix_op.linear_layer.weight"]
        model["transformer.h.3.mlp.c_proj.weight"] = state_dict["transformer.h.3.mlp.linear_1_mix_op.linear_layer.weight"]
        model["transformer.h.4.ln_1.weight"] = state_dict["transformer.h.4.ln1_op.layer_norm.weight"]
        model["transformer.h.4.ln_2.weight"] = state_dict["transformer.h.4.ln2_op.layer_norm.weight"]
        model["transformer.h.4.attn.c_attn.weight"] = state_dict["transformer.h.4.attn.c_attn_op.linear_layer.weight"]
        model["transformer.h.4.attn.c_proj.weight"] = state_dict["transformer.h.4.attn.c_proj_mix_op.linear_layer.weight"]
        model["transformer.h.4.ln_2.weight"] = state_dict["transformer.h.4.ln2_op.layer_norm.weight"]
        model["transformer.h.4.mlp.c_fc.weight"] = state_dict["transformer.h.4.mlp.linear_0_mix_op.linear_layer.weight"]
        model["transformer.h.4.mlp.c_proj.weight"] = state_dict["transformer.h.4.mlp.linear_1_mix_op.linear_layer.weight"]
        model["transformer.h.5.ln_1.weight"] = state_dict["transformer.h.5.ln1_op.layer_norm.weight"]
        model["transformer.h.5.ln_2.weight"] = state_dict["transformer.h.5.ln2_op.layer_norm.weight"]
        model["transformer.h.5.attn.c_attn.weight"] = state_dict["transformer.h.5.attn.c_attn_op.linear_layer.weight"]
        model["transformer.h.5.attn.c_proj.weight"] = state_dict["transformer.h.5.attn.c_proj_mix_op.linear_layer.weight"]
        model["transformer.h.5.ln_2.weight"] = state_dict["transformer.h.5.ln2_op.layer_norm.weight"]
        model["transformer.h.5.mlp.c_fc.weight"] = state_dict["transformer.h.5.mlp.linear_0_mix_op.linear_layer.weight"]
        model["transformer.h.5.mlp.c_proj.weight"] = state_dict["transformer.h.5.mlp.linear_1_mix_op.linear_layer.weight"]
        return model

        
    def optimize(self, n_iters: int = 100):
        """
        Run random search for n_iters function evaluations.
        """
        for i in range(n_iters):
            config = self.model.sample_random_config()
            self.train_and_eval(config,self.model,self.train_portion,self.optimizer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/work/dlclarge2/sukthank-tanglenas/merge/TangleNAS-dev/out_train_spos_spos_9004_0.8/ckpt.pt")
    parser.add_argument('--n_iters', type=int, default=5000)
    parser.add_argument('--train_portion', type=float, default=0.8)
    parser.add_argument('--optimizer', type=str, default="spos")
    args = parser.parse_args()
    with open('/work/dlclarge2/sukthank-tanglenas/merge/TangleNAS-dev/toy_search_spaces/nanoGPT/data/shakespeare_char/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    exp_name = args.model_path.split("/")[-2]
    # Checking all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    vocab_set = "".join(chars)
    # Create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    # Train and test splits
    train_size = 0.9
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_size * len(data))
    train_data = data[:n]
    valid_data = data[n:]
    train_portion = args.train_portion
    n_train = int(train_portion * len(train_data))
    #print(n_train)
    #print(len(train_data))
    train_data_split = train_data[:n_train]
    #print(len(train_data))
    eval_data_split = train_data[n_train:]
    #print(len(eval_data))
    model_args = dict(n_layer=[4,5,6], n_head=[2,4,6], n_embd=[96,192,384], block_size=256,
                  bias=False, vocab_size=65, dropout=0.2, mlp_ratio=[2,3,4])
    rs = RandomSearch(args.model_path, model_args, train_data_split, eval_data_split, valid_data, args.train_portion, args.optimizer, exp_name)
    rs.optimize(args.n_iters)