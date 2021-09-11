# Self-training using Transfer Learning
This repository contains the implementation of [Transfer of Pretrained Model Weights Substantially Improves Semi-Supervised 
Image Classification](https://arxiv.org/abs/2109.00788).


## How to run
For running  various experiments, `train.py` can be used. For setting up the environment, datasets preparation, 
and detail of the command-line arguments for running `train.py` can be found at [README.md](README.md).
For our experiments in the paper, VGG16 [1] with randomly initialised weights and ImageNet pretrained weights was used. 
The noteworthy is a boolean `weights` command-line parameter, which  by default is `False`. Passing `--weights` will use 
pretrained weights for the VGG16 network.   
For VGG16, publicly available pretrained weights from `tensorflow.keras` will be automatically downloaded (download 
size ~59 MB) and saved at `~/.keras/models/`. The  `VGG16` is used without classification head, with random and ImageNet
pretrained weights publically available from `tensorflow.keras`.

Alternatively, some sample configuration files are provided in the `config/` directory.

For example, one can run N-labelled (1000) training on SVHN using triplet loss [2] with random weights of VGG16 as:

` python train.py --flagfile config/svhn-vgg16-random-triplet-N-labelled.cfg`

For self-training on `cifar10` using cross-entropy loss with pretrained VGG16, we can run:

` python train.py --flagfile config/cifar10-vgg16-weights-cross-entropy-Self-training.cfg`

For self-training on `plant96` using ArcFace loss [3] with pretrained VGG16, we can run:

` python train.py --flagfile config/plant96-vgg16-weights-arcface-Self-training.cfg`

For All-labelled on `plant96` using Contrastive loss [3] with randomly initialised VGG16, we can run:

` python train.py --flagfile config/plant96-vgg16-random-contrastive-All-labelled.cfg`


### Training logs

Training logs will be saved at `{loss_type}_logs/dataset/network/`. The loss type can be `triplet, contrastive, arcface, 
and cross-entropy`. The value for the dataset parameter can be `svhn_cropped, cifar10, and plant96`. 



## Citation Information

```
@inproceedings{sahito2020transfer,
  title={Transfer of Pretrained Model Weights Substantially Improves Semi-supervised Image Classification},
  author={Sahito, Attaullah and Frank, Eibe and Pfahringer, Bernhard},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={433--444},
  year={2020},
  organization={Springer}
}
```

## References
1. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." 
arXiv preprint arXiv:1409.1556 (2014).
2. Distance metric learning for large margin nearest neighbour classification. Kilian Q Weinberger and Lawrence K Saul.
Journal of Machine Learning Research,  10(2), 2009.
3. Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." Proceedings of the 
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
4. Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." 2006 IEEE 
Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). Vol. 2. IEEE, 2006.

