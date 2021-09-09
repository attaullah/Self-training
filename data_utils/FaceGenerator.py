import tensorflow.keras
import numpy as np


class FaceDataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, x_train, y_train, batch_size=32, shuffle=True, augmentations=None):
        """
        tf.keras sequence based data generator used for training ArcFace loss.
        :param x_train: training images
        :param y_train: labels of training images
        :param batch_size: size of mini-batch
        :param shuffle: shuffle data
        :param augmentations: augmentations type of tf.keras.preprocessing.image.ImageDataGenerator
        """
        self.batch_size = batch_size
        self.X_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.X_train) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: of batches
        :return: images and labels for a batch
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.X_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        x = self.X_train[batch_ids]
        y = self.y_train[batch_ids]
        if self.augmentations:
            for i in range(self.batch_size):
                x[i] = self.augmentations.random_transform(x[i])
        return [x, y], y

