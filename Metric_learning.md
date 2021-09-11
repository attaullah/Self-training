# Self-training using Metric learning
This repository contains the implementation of [Semi-supervised Learning Using Siamese
Networks](https://arxiv.org/abs/2109.00794).

## How to run
For running  various experiments, `train.py` can be used. For setting up the environment, datasets preparation, 
and detail of the command-line arguments for running `train.py` can be found at [README.md](README.md).

Alternatively, some sample configuration files are provided in the `config/` directory.
For example, one can run all-labelled (60000) training on MNIST using the `simple` custom convolutional model by:

` python train.py --flagfile config/mnist-simple-cross-entropy-All-labelled.cfg`

Triplet loss [1] based self-training on CIFAR-10 using `ssdl` convolutional model and `1-nn` as a confidence measure for 
selection of pseudo-labels:

` python train.py --flagfile config/cifar10-ssdl-arcface-Self-training-1-nn.cfg`

For Fashion-MNIST, self-training experiment using the `simple` custom convolutional model with `triplet` loss and 
LLGC [2] as a confidence measure for selection of pseudo-labels:

` python train.py --flagfile config/fashion_mnist-simple-triplet-Self-training-llgc.cfg`

For N-labelled (1000) training on SVHN using the `simple` network model with cross-entropy loss, we can run:

` python train.py --flagfile config/svhn-simple-cross-entropy-N-labelled.cfg`

### Training logs

Training logs with all the parameter details and test accuracies will be saved at `{loss_type}_logs/dataset/network/`. The loss type can be `triplet and cross-entropy`. 
The value for dataset parameter can be `mnist, fashion_mnist, svhn_cropped, and cifar10`. 
The `simple` custom network model is used for MNIST, Fashion-MNIST, and SVHN, while for CIFAR-10, `ssdl` is used.

## Citation info

```
@inproceedings{sahito2019semi,
  title={Semi-supervised learning using Siamese networks},
  author={Sahito, Attaullah and Frank, Eibe and Pfahringer, Bernhard},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={586--597},
  year={2019},
  organization={Springer}
}
```
## References
1. Distance metric learning for large margin nearest neighbour classification. Kilian Q Weinberger and Lawrence K Saul.
Journal of Machine Learning Research,  10(2), 2009.
2. Zhou, Dengyong, et al. "Learning with local and global consistency." Advances in neural information processing 
systems. 2004.
