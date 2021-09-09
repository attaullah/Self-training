import numpy as np


class DataSet(object):

    def __init__(self, images, labels, one_hot=False, n_classes=10, shuffle=True):
        """
        Create a dataset object based on images and labels and support batching.
        :param images: input images
        :param labels: input labels
        :param one_hot: whether to apply one_hot encoding to labels or not
        :param n_classes: number of classes in dataset
        :param shuffle: shuffle dataset
        """
        assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
        self._num_examples = images.shape[0]
        if shuffle:
            perm = np.arange(self._num_examples)
            shuffled_indices = np.random.permutation(perm)
            np.random.shuffle(shuffled_indices)
            images = images[shuffled_indices]
            labels = labels[shuffled_indices]

        self._images = images
        if one_hot:
            labels = np.eye(n_classes)[labels]
        self._labels = labels

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled, one_hot, n_classes=10, shuffle=False):
        """
        Semi-supervised dataset object for handling labelled and unlabelled examples and batches.
        :param images: input images
        :param labels: input labels
        :param n_labeled: number of examples selected as labelled and rest are considered as unlabelled examples
        :param one_hot: whether to apply one_hot encoding to labels or not
        :param n_classes: Number of classes
        :param shuffle: shuffle dataset
        """
        self.n_labeled = n_labeled
        self.num_examples = len(images)

        if shuffle:
            perm = np.arange(len(images))
            shuffled_indices = np.random.permutation(perm)
            np.random.shuffle(shuffled_indices)
            images = images[shuffled_indices]
            labels = labels[shuffled_indices]

        lab_inds = []
        n_from_each_class = int(n_labeled / n_classes)  # select same number of labelled examples from each class
        for c in range(n_classes):
            lab_inds += [i for i, cl in enumerate(labels) if c == cl][:n_from_each_class]
        l_images = images[lab_inds]
        l_labels = labels[lab_inds]
        l_labels = l_labels.reshape(l_labels.shape[0])
        # Labelled dataset object
        self.labeled_ds = DataSet(l_images, l_labels, one_hot=one_hot, n_classes=n_classes, shuffle=True)

        # Unlabeled DataSet
        unlabeled_imgs = np.delete(images, lab_inds, 0)
        unlabeled_lbls = np.delete(labels, lab_inds, 0)  # using original labels for accuracy calculation
        self.unlabeled_ds = DataSet(unlabeled_imgs, unlabeled_lbls, one_hot=one_hot, n_classes=n_classes,
                                    shuffle=shuffle)

    def next_batch(self, batch_size):
        # ignoring unlabelled labels
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = np.vstack([labeled_images, unlabeled_images])
        return images, labels
