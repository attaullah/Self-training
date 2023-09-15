# Self-training using Self-supervised Learning
This repository contains the implementation of [Better Self-training for Image Classification through 
Self-supervision](https://arxiv.org/abs/2109.00778).


## How to run
For setting up the environment, datasets preparation and detail of the command-line arguments can be found at 
[README.md](README.md). For supervised training, `loss_type` can be one of `cross-entropy` or `triplet` [1]. For 
self-supervision, the cross-entropy loss is used on auxiliary tasks based on rotations [2] and flips [3]. 

### Pretrained weights
The Wide Residual Network (WRN) [4] is used as a base model for supervised and self-supervised training. 
Specifically, WRN-28-2 with randomly initialised and ImageNet 32x32 pretrained weights are used. For  details about 
downloading pretrained weights, kindly visit
[Pretraining-WideResNet](https://github.com/attaullah/Pretraining-WideResNet). The pretrained weights 
`32x32-CE-weights.h5` (download size ~6MB) can be downloaded. For the pretrained version, the WRN-28-2 network model expects 
pretrained weights saved at `weights/` directory. 

For using pretrained weights  `--weights` command-line parameter should be passed to the training script.

For various experiments, `SS_pretrain.py` and `SS_train.py` can be used with suitable command-line parameters. 
For self-training, self-supervision is applied in three different ways. They are described as follows:

### 1- Self-supervised Pretraining
`SS_pretrain.py` introduces  one extra parameter for the number of pretraining epochs `pretraining_epochs` default set to 200.

For N-labelled (4000) training on `cifar10` using triplet loss with randomly initialised weights of  WRN-28-2, we can 
run:

`python SS_pretrain.py --flagfile config/SS/pretrain-cifar10-wrn-28-2-random-triplet-N-labelled.cfg`

For self-training on `plant64` using  cross-entropy loss with pretrained WRN-28-2, we can run:

`python SS_pretrain.py --flagfile config/SS/pretrain-plant64-wrn-28-2-weights-cross-entropy-Self-training.cfg`

For All-labelled training on `plant32` using  cross-entropy loss with pretrained WRN-28-2, we can run:

`python SS_pretrain.py --flagfile config/SS/pretrain-plant32-wrn-28-2-weights-cross-entropy-All-labelled.cfg`

### 2- Combined Training (CT)
`SS_train.py` offers two extra parameters: `lambda_u` (weight of self-supervised loss) default set to 1 and 
self-training type `st_type`. For applying self-supervised and supervised, i.e., combined training in all iterations of 
self-training `st_type` to `combined`.

For N-labelled `Combined Training` on `plant96` using cross-entropy loss  with pretrained WRN-28-2, we can run:

`python SS_train.py --flagfile config/SS/plant96-wrn-28-2-weights-cross-entropy-N-labelled-combined.cfg`

For Self-training using `Combined Training` on `cifar10` using cross-entropy loss  with randomly initialised weights of 
WRN-28-2, we can run:

`python SS_train.py --flagfile config/SS/cifar10-wrn-28-2-random-cross-entropy-Self-training-combined.cfg`

For Self-training using `Combined Training` on `plant64` using triplet loss  with pretrained WRN-28-2, we can run:

`python SS_train.py --flagfile config/SS/plant64-wrn-28-2-weights-triplet-Self-training-combined.cfg`


For All-labelled `Combined Training` on `plant32` using triplet loss  with randomly initialised weights of 
WRN-28-2, we can run:

`python SS_train.py --flagfile config/SS/plant32-wrn-28-2-random-triplet-All-labelled-combined.cfg.cfg`

### 3- Self-training Single-Step Combined training (STSSC)
STSSC is a special type of Combined Training (CT) for self-training. Combined training of self-supervised and 
supervised training is applied only to the first iteration, and subsequent iteration of self-training uses only 
supervised training. STSSC can be run by setting self-training type `--st_type stssc`.

For STSSC on `svhn_cropped` using triplet loss with pretrained WRN-28-2, we can run:

`python SS_train.py --flagfile config/SS/svhn-wrn-28-2-weights-triplet-Self-training-stssc.cfg`

For STSSC on `plant96` using cross-entropy loss with randomly initialised weights of WRN-28-2, we can run:

` python SS_train.py --flagfile config/SS/plant96-wrn-28-2-random-cross-entropy-Self-training-stssc.cfg`


## Training logs
For self-supervised pretraining, logs will be saved at

`ss-pretrain-{loss_type}_logs/dataset/wrn-28-2/`. 

For CT and STSSC, logs will be saved at  

`ss-pretrain-{loss_type}_logs/dataset/wrn-28-2/`

The values for the dataset can be one of `cifar10`,`svhn_cropped`,`plant32`, `plant64`, and `plant96`. 

## Citation information
```
@InProceedings{10.1007/978-3-030-97546-3_52,
author="Sahito, Attaullah
and Frank, Eibe
and Pfahringer, Bernhard",
editor="Long, Guodong
and Yu, Xinghuo
and Wang, Sen",
title="Better Self-training for Image Classification Through Self-supervision",
booktitle="AI 2021: Advances in Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="645--657",
}

```
## References
1. Distance metric learning for large margin nearest neighbour classification. Kilian Q Weinberger and Lawrence K Saul.
Journal of Machine Learning Research,  10(2), 2009.
2. Komodakis, Nikos, and Spyros Gidaris. "Unsupervised representation learning by predicting image rotations." 
International Conference on Learning Representations (ICLR). 2018.
3. Tran, Phi Vu. "Exploring self-supervised regularization for supervised and semi-supervised learning." arXiv preprint 
arXiv:1906.10343 (2019).
4. Wide Residual Networks. Sergey Zagoruyko and Nikos Komodakis. In British
Machine Vision Conference 2016. British Machine Vision Association, 2016.
