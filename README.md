# Self-training
This repository contains code for PhD thesis: "A Study of Self-training Variants for
Semi-supervised Image Classification" (under examination) and publications.
1. Sahito A., Frank E., Pfahringer B. (2019) Semi-supervised Learning Using Siamese Networks. In: Liu J., Bailey J. 
(eds) AI 2019: Advances in Artificial Intelligence. AI 2019 . Lecture Notes in Computer Science, vol 11919. Springer, 
Cham. [DOI:978-3-030-35288-2_47](https://link.springer.com/chapter/10.1007/978-3-030-35288-2_47) 
2. Sahito A., Frank E., Pfahringer B. (2020) Transfer of Pretrained Model Weights Substantially Improves Semi-supervised
Image Classification. In: Gallagher M., Moustafa N., Lakshika E. (eds) AI 2020: Advances in Artificial Intelligence.
AI 2020 . Lecture Notes in Computer Science, vol 12576. Springer, Cham. 
[DOI:978-3-030-64984-5_34](https://doi.org/10.1007/978-3-030-64984-5_34)
3. Sahito A., Frank E., Pfahringer B. (2021) Better Self-training for Image Classification through Self-supervision. 
[arXiv:2109.00778](https://arxiv.org/abs/2109.00778)

## Getting started
Start with cloning the repo:
```bash
git clone https://github.com/attaullah/self-training.git
cd self-training/
```
### Environment setup
For creating conda environment, a yml  file `tf.yml` is provided for replicating setup.

```bash
conda env create -f tf.yml
conda activate tf
```

### Data preparation
MNIST, Fashion-MNIST, SVHN, and CIFAR-10 datasets are loaded using   [TensorFlow  datasets](https://www.tensorflow.org/datasets). 
package, which can be installed using pip:
```bash
pip install tensorflow_datasets
```
By default, tensorflow_datasets package will save datasets at `~/ tensorflow_datasets/` directory.

For PlantVillage dataset please follow instructions at
 [plant-disease-dataset](https://github.com/attaullah/downsampled-plant-disease-dataset). The downloaded files are 
expected to be saved in `data/` directory. 

### Pretrained weights
For details about downloading pretrained weights, see [Trasnfer_learning](Transfer_learning.md).

### Repository Structure
Here is the brief overview of the repository.

-`config/`: contains sample configurations file for running various experiments.

-`data_utils/`: provides helper functions for loading datasets, details of  datasets like number of initially labelled
examples: `n_label`, selection percentage of pseudo-labels for each iteration of self-training: `selection_percentile`,
parameter `sigma` for LLGC and `tensorflow.keras` based dataloaders for training the network model.

-`losses/`: implementation of ArcFace, Contrastive, and Triplet loss.

-`models/`: provides implementation of custom `SIMPLE` convolutional network model used for MNIST, Fashion-MNIST, and 
SVHN, `SSDL` and other custom convolutional network model used for CIFAR-10 and PlantVillages.

-`utils/`: contains implementation of LLGC and other utility function.


## Example usage
Training can be started using `train.py` script. Details of self-explanatory commandline 
arguments can be seen by passing `--helpfull` to it.


```bash
 python train.py --helpfull
 
       USAGE: pretrain.py [flags]
flags:

  --dataset: <cifar10|mnist|fashion_mnist|svhn_cropped|plant32|plant64|plant96>: dataset name
    (default: 'cifar10')
  --network: <wrn-28-2|simple|vgg16|ssdl>: network architecture.
    (default: 'wrn-28-2')
  --[no]weights: random or ImageNet pretrained weights
  --batch_size: size of mini-batch
    (default: '100')
    (an integer)
  --epochs: initial training epochs
    (default: '200')
    (an integer)
  --[no]semi: True: N-labelled training, False: All-labelled training
      (default: 'true')
  --lt: <cross-entropy|triplet|arcface|contrastive>: loss_type: cross-entropy, triplet,  arcface or contrastive.
    (default: 'cross-entropy')
  --opt: <adam|sgd|rmsprop>: optimizer.
   (default: 'adam')
  --lr: learning_rate
    (default: '0.0001')
    (a number)
  --lbl: <knn|lda|rf|lr>: shallow classifiers used for test accuracy forr metric learning losses
    (default: 'knn')
  --margin: margin for triplet loss calculation
    (default: '1.0')
    (a number)
 
  --[no]self_training: apply self-training
    (default: 'false')
  --confidence_measure: <1-nn|llgc>: confidence measure for pseudo-label selection.
    (default: '1-nn')
 --meta_iterations: number of self-training meta_iterations
    (default: '25')
    (an integer)
  --epochs_per_m_iteration: number of epochs per meta-iteration
    (default: '200')
    (an integer)
    
  --gpu: gpu id
    (default: '0')
  --pre: prefix for log directory
    (default: '')
  --verbose: verbose level 1 or 0.
    (default: '1')
    (an integer)
 ```
For running different experiments of self-training based on metric learning losses, visit
[Metric-learning](Metric_learning.md) and for self-supervised,
[Self-supervised](Self_supervised.md)
## Citation 
If you use the provided code, kindly cite our paper.
```
@misc{sahito2021better,
      title={Better Self-training for Image Classification through Self-supervision}, 
      author={Attaullah Sahito and Eibe Frank and Bernhard Pfahringer},
      year={2021},
      eprint={2109.00778},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## References
1. Wide Residual Networks. Sergey Zagoruyko and Nikos Komodakis. In British
Machine Vision Conference 2016. British Machine Vision Association, 2016.
3. Distance metric learning for large margin nearest neighbour classification. Kilian Q Weinberger and Lawrence K Saul.
Journal of Machine Learning Research,  10(2), 2009.
4. Zhou, Dengyong, et al. "Learning with local and global consistency." Advances in neural information processing 
systems. 2004.
5. J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep 
Convolutional Neural Network”, Mendeley Data, V1, doi: 10.17632/tywbtsjrjv.1


## License
[MIT](https://choosealicense.com/licenses/mit/)