"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_base import GPTConfig, GPT
import argparse

from search_spaces.NB201.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=9001, help="random seed")
parser.add_argument(
    "--arch_traj_load_path",
    type=str,
    default=None,
    help="Path to load the architecture trajectory from. Use this argument to train the best model found in a one-shot search.",
)
parser.add_argument(
    "--arch_config_file",
    type=str,
    default=None,
    help="Path to load the architecture config from. Use this argument to train a custom model. Either this argument or arch_traj_load_path argument must be specified, but not both.",
)
parser.add_argument(
    "--data_dir", type=str, default="search_spaces/nanoGPT/data/tinystories", help="Path to the data directory."
)
parser.add_argument(
    "--early_stopping", action="store_true", help="Enable early stopping."
)
parser.add_argument(
    "--val_portion",
    type=float,
    default=0.,
    help="Portion of the train data to use as validation. Used only when --early_stopping flag is used.",
)
parser.add_argument(
    "--max_iters",
    type=int,
    required=True,
    help="Maximum number of iterations to train the model. If --early_stopping flag is used, training might stop before max_iters",
)
args, unknown_args = parser.parse_known_args()
print(args)

assert (args.arch_traj_load_path is None) ^ (
    args.arch_config_file is None
), "Exactly one of {arch_traj_load_path, arch_config_file} must be specified"

if args.early_stopping is True:
    assert (
        args.val_portion > 0.0
    ), "Validation portion must be > 0. when using early stopping"

if args.arch_traj_load_path is not None:
    search_train_portion = args.arch_traj_load_path.split("_")[-4]
    search_seed = args.arch_traj_load_path.split("_")[-3]
    search_max_iters = args.arch_traj_load_path.split("_")[-2]
    search_optimizer = args.arch_traj_load_path.split("_")[2]

    if search_optimizer == "darts":
        search_optimizer = search_optimizer + args.arch_traj_load_path.split("_")[3]

    with open(args.arch_traj_load_path, "rb") as f:
        arch_traj = pickle.load(f)
    for k in arch_traj.keys():
        config = arch_traj[k]
else:
    search_train_portion = None
    search_seed = None
    search_max_iters = None
    search_optimizer = None

    config = load_config(args.arch_config_file, None, None)._asdict()

print(config)
config_string = ""
for k, v in config.items():
    k_ = "".join(map(lambda s: s[0], k.split("_")))
    v_ = "".join(map(str, v)) if isinstance(v, list) else str(v)
    config_string += f"{k_}{v_}_"
config_string = config_string[:-1]

n_layer = config["num_layers"]
n_embd = config["embed_dim"]
n_heads = config["num_heads"]
mlp_ratio = config["mlp_ratio"]
# get time string
timestr = time.strftime("%Y%m%d-%H%M%S")
out_dir = f"output_tinystories/out_train_{search_optimizer}_{search_train_portion}_{config_string}_maxiter{args.max_iters}_{args.seed}_{timestr}"
eval_interval = 2000
log_interval = 1
eval_iters = 10
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())

# data
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
# lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.

# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(
    open("search_spaces/nanoGPT/configurator.py").read()
)  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

print(f"Checkpoint files will be saved to {out_dir}")

torch.manual_seed(args.seed + seed_offset)
np.random.seed(args.seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
train_data_full = np.memmap(
    os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r"
)

# Get the train/validation split
# Note: The rows are not shuffled
train_ratio = 1 - args.val_portion
split_idx = int(train_ratio * len(train_data_full))

train_data = train_data_full[:split_idx]
val_data = train_data_full[split_idx:]
test_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

def get_batch(split):
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    elif split == "test":
        data = test_data
    else:
        raise ValueError(f"split{split} not recognized")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(args.data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_heads,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    mlp_ratio=mlp_ratio,
)  # start with model_args from command line

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "latest_ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even: training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    """for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model_args['bias'] = False"""  # override bias from command line
    # create the model
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    print(f"model_args: {model_args}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    # for k,v in list(state_dict.items()):
    #    print(k,v)
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args[
        "block_size"
    ] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(splits):
    out = {}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


lr_decay_iters = max_iters # Keeping it simple

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb_run_name = f"TinyStories_trainmodel_{search_optimizer}_{search_train_portion}_{config_string}_{args.max_iters}_{args.seed}"
    wandb.init(project=wandb_project, name=wandb_run_name, entity='', config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

splits = ["train", "val"] if args.val_portion > 0.0 else ["train"]

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    is_last_iter = iter_num == max_iters

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 or is_last_iter) and master_process:
        losses = estimate_loss(splits)
        val_loss = losses["val"] if "val" in losses else 1e9
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"] if "val" in losses else 1e9,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "best_val_loss": best_val_loss,
                }
            )

        if (
            ("val" in losses and losses["val"] < best_val_loss)
            or always_save_checkpoint
            or is_last_iter
        ):
            best_val_loss = (
                losses["val"]
                if ("val" in losses and losses["val"] < best_val_loss)
                else best_val_loss
            )
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")

                if "val" in losses and losses["val"] < best_val_loss:
                    torch.save(checkpoint, os.path.join(out_dir, "best_ckpt.pt"))

                torch.save(checkpoint, os.path.join(out_dir, "latest_ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

test_loss = estimate_loss(["test"])["test"]

print(f"Training complete. Test loss: {test_loss}")

if wandb_log and master_process:
    wandb.log(
        {
            "iter": iter_num,
            "test/loss": test_loss,
            "lr": lr,
            "mfu": running_mfu * 100,  # convert to percentage
            "best_val_loss": best_val_loss,
        }
    )

if ddp:
    destroy_process_group()
