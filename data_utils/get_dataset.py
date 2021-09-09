import numpy as np
import tensorflow_datasets as tfds
from .plant_village import load_dataset
from .dataset import DataSet, SemiDataSet
from .dataset_config import data_details


def read_data_sets(name, one_hot=False, semi=True, scale=True, shuffle=True):
    """
    Prepare the dataset. The mnist, fashion_mnist, svhn_cropped are  cifar10 are loaded through tensorflow_datasets
    package, while for plant** datasets, visit https://github.com/attaullah/downsampled-plant-disease-dataset.
    :param name: name of dataset. One of mnist, fashion_mnist, svhn_cropped, cifar10, plant32, plant64, and plant96
    :param one_hot: use one-hot encoding
    :param semi: semi=True means N-labelled and semi=False means All-labelled
    :param scale: to perform scaling... *1/255.
    :param shuffle: shuffle data
    :return: dataset object containing training and test datasets and dataset details
    """
    class DataSets(object):
        pass
    data_sets = DataSets()

    if 'plant' in name:
        train_images, train_labels, test_images, test_labels = load_dataset(name)
        n_classes = 38
    else:
        n_classes = 10
        ds_train, ds_test = tfds.load(name=name, split=["train", "test"], batch_size=-1,)
        # convert tfds dataset to numpy arrays and change dtypes
        ds_train_n = tfds.as_numpy(ds_train)
        ds_test_n = tfds.as_numpy(ds_test)
        train_images, train_labels = ds_train_n["image"], ds_train_n["label"]
        test_images, test_labels = ds_test_n["image"], ds_test_n["label"]

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    if scale:
        test_images = np.multiply(test_images, 1.0 / 255.0)
        train_images = np.multiply(train_images, 1.0 / 255.0)

    n_labeled, selection_percentile, sigma = data_details(name)
    if semi:
        data_sets.train = SemiDataSet(train_images, train_labels, n_labeled, one_hot=one_hot, n_classes=n_classes,
                                      shuffle=shuffle)
    else:
        n_labeled = train_images.shape[0]  # in case of all-labelled examples
        data_sets.train = DataSet(train_images, train_labels, one_hot=one_hot, shuffle=shuffle)
    data_sets.test = DataSet(test_images, test_labels, one_hot=one_hot, shuffle=shuffle, n_classes=n_classes)

    class Config(object):
        pass
    # dataset attributes
    data_config = Config()
    data_config.name = name
    data_config.channels = train_images.shape[-1]
    data_config.size = train_images.shape[1]
    data_config.nc = n_classes
    data_config.n_label = n_labeled
    data_config.sp = selection_percentile
    data_config.sigma = sigma
    data_config.semi = semi

    return data_sets, data_config
