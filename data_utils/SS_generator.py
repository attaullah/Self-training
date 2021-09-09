import numpy as np


def geometric_transform(image, auxiliary_labels=6, one_hot=True):
    """
    Applies six transformations: rotation by 0,90, 180, 270 and flip left-right and flip upside-down.
    :param image: input image
    :param auxiliary_labels: number of auxiliary labels
    :param one_hot: apply one-hot encoding to labels
    :return: six images with all six transformations applied and auxiliary labels 0...auxiliary_labels-1
    """
    _, h, w, c = image.shape
    image = np.reshape(image, (h, w, c))
    labels = np.empty((auxiliary_labels,), dtype='uint8')
    images = np.empty((auxiliary_labels, h, w, c), dtype='float32')
    for i in range(auxiliary_labels):
        if i <= 3:
            t = np.rot90(image, i)
        elif i == 4:
            t = np.fliplr(image)
        else:
            t = np.flipud(image)
        images[i] = t
        labels[i] = i
    if one_hot:
        return images, np.eye(auxiliary_labels)[labels]
    else:
        # print('one_hot = ', one_hot, labels.shape)
        labels = labels.squeeze()
        # print('one_hot = ', one_hot, labels.shape)
        return images, labels


def combined_generator(super_iter, self_iter, batch_size, one_hot=True):
    """
    Utility function to load data into required Keras model format for training on supervised batch and self-supervised
    batch.
    :param super_iter: supervised data generator based on labelled training images
    :param self_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    """
    super_batch = batch_size * 1
    self_batch = batch_size
    while True:
        x_super, y_super = zip(*[next(super_iter) for _ in range(super_batch)])
        x_self, y_self = zip(*[geometric_transform(next(self_iter), one_hot=one_hot)
                               for _ in range(self_batch)])

        x_super = np.vstack(x_super)
        y_super = np.vstack(y_super)
        x_self = np.vstack(x_self)
        y_self = np.vstack(y_self)
        if not one_hot:
            y_self = y_self.ravel()
        yield [x_self, x_super], [y_self, y_super]


def self_supervised_data_generator(self_iter, batch_size, one_hot=True):
    """
    Utility function to load data into required Keras model format.
    :param self_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    """
    self_batch = batch_size
    while True:
        x_self, y_self = zip(*[geometric_transform(next(self_iter), one_hot=one_hot)
                               for _ in range(self_batch)])
        x_self = np.vstack(x_self)
        if not one_hot:
            y_self = np.array(y_self)
            y_self = y_self.ravel()
        y_self = np.vstack(y_self)
        yield x_self, y_self

