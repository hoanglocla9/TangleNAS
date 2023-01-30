# TangleNAS
Official repo for the paper TangleNAS: Weight Entanglement meets One-Shot Optimization.

![title](figures/overview.png)
## Overview
1. [Installation & Dependencies](#Dependencies)
2. [Prepration](#Prepration)
    1. [Directories](#Directories)
    2. [Data](#Data)
3. [Training](#Training)
    1. [Searching + Training and Evaluation](#launch)
4. [Results](#Results)

## 1. Installation & Dependencies<a name="Dependencies"></a>


To install the dependencies:
```bash
conda create --name tanglenas python=3.9
conda activate tanglenas
pip install -r requirements.txt
```

## 2. Preparation <a name="Preparation"></a>
```bash
├── configs
│   ├── finetune
│   ├── search
│   └── train
├── job_scripts
├── optimizers
│   ├── mixop
│   └── sampler
│   ├── optim_factory.py
├── plotting
├── search_spaces
│   ├── AutoFormer
│   ├── DARTS
│   ├── NB201
│   ├── NATS
│   ├── SwinIR
├── search
├── toy_search_space
├── train
```

The ```configs``` folder contains the configs used for search, finetuning and training the architectures obtained

The ```job_scripts``` folder contains the scripts used for search, finetuning and training the architectures obtained

The ```optimizers``` folder contains the configuratble optimizers used namely darts, drnas and gdas

The ```job_scripts``` folder contains the scripts used for search, finetuning and training the architectures obtained

The ```plotting``` folder contains the scripts used for analysis

The ```search_spaces``` folder contains the definition of the search spaces used

The ```search``` folder contains the code used to perform NAS

The ```train``` folder contains the code used to train or finetune the architectures obtained

The ```toy_search_space``` folder contains the code for the search spaces and optimizers of the toy search space.




