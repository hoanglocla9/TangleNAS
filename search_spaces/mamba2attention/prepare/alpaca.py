# Taken from Karpathy's NanoGPT repository (https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py)
# and modified to work with alpaca

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
from transformers import GPTNeoXTokenizerFast
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 16

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    dataset = load_dataset("tatsu-lab/alpaca", num_proc=num_proc_load_dataset)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    # enc = tiktoken.get_encoding("gpt2")
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" #enforce padding side left

    def process(example):
        ids = tokenizer.encode(example['text'])  # use tokenizer to encode
        ids.append(tokenizer.eos_token_id)  # add the end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    alpaca_datapath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "alpaca"
    )

    if not os.path.exists(alpaca_datapath):
        os.makedirs(alpaca_datapath)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(alpaca_datapath, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()