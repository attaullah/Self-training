# Pretrained weights

For our experiments, we only use pretrained weights for VGG16 and Wide Residual Network.

## VGG16
For VGG16, publicly available pretrained weights from `tensorflow.keras` will be automatically downloaded and saved at 
`~/.keras/models/`.

## Wide Residual Network
 For Wide 
Residual Network (WRN), we pretrain WRN-28-2 on ImageNet 32x32 using cross-entropy loss. For more details, kindly visit
[Pretraining-WideResNet](https://github.com/attaullah/Pretraining-WideResNet). The pretrained weights 
`32x32-CE-weights.h5` can be downloaded from the above mentioned repository. WRN-28-2 network model expects pretrained 
weights saved at `weights/` directory.

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