# One-Shot NanoGPT Search with TinyStories

## Install dependencies
1. pip install tiktoken datasets

## Prepare the data
1. Run `python data/prepare/tinystories` from the root directory of the project.
2. Copy the `train.bin` and `validation.bin` to `toy_search_spaces/nanoGPT/data/tinystories`
3. Rename `validation.bin` to `val.bin` in `toy_search_spaces/nanoGPT/data/tinystories`

## One-Shot Search on the NanoGPT supernet (DrNAS/SPOS)

The configuration for search is taken from `toy_search_spaces/nanoGPT/config/train_tinystories_{drnas/spos}.py`

To run the search:

```python toy_search_spaces/nanoGPT/train_search.py config=toy_search_spaces/nanoGPT/config/train_tinystories_{drnas/spos}.py```

The job script (DDP, with 4 GPUs) can be found in `scripts/train_nanogpt_{drnas/spos}.sh`


## Training a model from scratch

There are two ways to specify the config of the model to train from scratch:
1. Load it from the arch trajectory pickle file of a previous one-shot search
2. Load it from a custom arch config file

You can use either one option when running `train_base.py`, but not both (it will throw an assertion error if you attempt to do this).

### 1. Load from arch trajectory
`python toy_search_spaces/nanoGPT/train_base.py config=toy_search_spaces/nanoGPT/config/train_tinystories_base.py --arch_traj_load_path output_tinystories/out_search_drnas_0.5_42_7500_20230824-182026 --max_iters=12000`

The directory is of the format out_search_<optimizer>_<train_portion>_<seed>_<max_iters>_<formatted_time>z

### 2. Load from config file
`python toy_search_spaces/nanoGPT/train_base.py config=toy_search_spaces/nanoGPT/config/train_tinystories_base.py --arch_config_file=toy_search_spaces/nanoGPT/config/nanoGPT_train_ts.config --max_iters=12000`

When training from scratch, you can optionally provide a `--val_portion` argument, which will tell the script to split the train data into train and validation splits. By default, this split is disabled, and the validation loss in the logs will be entered as 1e9.