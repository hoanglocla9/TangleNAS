# TangleNAS
Official repo for the paper TangleNAS: Weight Entanglement meets One-Shot Optimization.

![title](figures/overview.png)
## Overview
1. [Installation & Dependencies](#Dependencies)
2. [Working Tree and Dataset Preparation](#dataset)
3. [Experiments](#experiments)
    - [Search](#search)
    - [Training and Evaluation](#launch)


## 1. Installation & Dependencies<a name="Dependencies"></a>


To install the dependencies:
```bash
conda create --name tanglenas python=3.9
conda activate tanglenas
pip install -r requirements.txt
```

## 2. Working Tree and Dataset Preparation <a name="dataset"></a>
### Code working tree
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

### Dataset preparation

```CIFAR10``` and ```CIFAR100``` datasets will be automatically downloaded
Download the ```imagenet-1k``` from [here](https://www.image-net.org/download.php) and update the path to the dataset in the training script. The dataset Imagenet16-120 

Download the DIV2K datasets from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and the corresponding Set5 and Set14 testsets from [here](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#Common-Image-SR-Datasets).

## 3. Experiments <a name="experiments"></a>

### Search  <a name="search"></a>
### Search on the NB201 search space
```bash
source job_scripts/launch_nb201_cifar.sh
```

```bash 
source job_scripts/launch_nb201_imgnet.sh
```

### Search on the NATS search space
```bash
source job_scripts/launch_nats_v2_cifar.sh
```

```bash
source job_scripts/launch_nats_v2_imgnet.sh
```

### Search on the DARTS search space

```bash
source job_scripts/launch_darts_cifar10.sh
```

```bash
source job_scripts/launch_darts_drnas.sh
```

### Search on the AutoFormer search space
```bash
source job_scripts/job_swinir_search.sh
```
### Search on the SwinIR search space
```bash
source job_scripts/job_swinir_search.sh
```

### Search on the Autoformer search space
```bash
source job_scripts/job_autoformer_search.sh
```










