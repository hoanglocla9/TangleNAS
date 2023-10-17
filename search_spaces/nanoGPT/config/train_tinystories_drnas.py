# Train NanoGPT on TinyStories
# Default settings taken from Karpathy's llama2.c repository
# https://github.com/karpathy/llama2.c/blob/ee95b1bf2943f87bc3e2c845bf07a733aca3806b/train.py

eval_interval = 100
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'tinystories-gpt2enc'
wandb_run_name = 'nano-gpt'

gradient_accumulation_steps = 5 * 4 # 5 steps per GPU, 4 GPUs
batch_size = 12
block_size = 224

# baby GPT model :)
n_layer = [5, 6, 7]
n_heads = [6, 8, 12]
n_embd = [384, 576, 768] # for gpt2, 768 is the largest
mlp_ratio = [2, 3, 4]
dropout = 0.2

learning_rate = 5e-4 # with baby networks can afford to go a bit higher
min_lr = 5e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

max_iters = 6000
lr_decay_iters = 6000
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
