import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D


class Simple(tf.keras.Model):
    def __init__(self, size, channels):
        """
        Simple custom keras model used for mnist, fashion_mnist, and svhn_cropped
        :param size: size of an image
        :param channels: number of channels in an image
        """
        super(Simple, self).__init__()
        self.conv1 = Conv2D(32, 7, input_shape=(size, size, channels), padding='same', activation='relu')
        self.mp = MaxPool2D(strides=2, padding='same', )
        self.conv2 = Conv2D(64, 5, padding='same', activation='relu')
        self.mp1 = MaxPool2D(strides=2, padding='same', )
        self.conv3 = Conv2D(128, 3, padding='same', activation='relu')
        self.mp2 = MaxPool2D(strides=2, padding='same', )
        self.conv4 = Conv2D(256, 1, padding='same', activation='relu')
        self.mp3 = MaxPool2D(strides=2, padding='same', )
        self.conv5 = Conv2D(4, 1, padding='same', activation='linear')
        self.flatten = Flatten()

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp1(x)
        x = self.conv3(x)
        x = self.mp2(x)
        x = self.conv4(x)
        x = self.mp3(x)
        x = self.conv5(x)
        x = self.flatten(x)

        return x
