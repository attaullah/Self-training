import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, GlobalAvgPool2D


class SSDL(tf.keras.Model):
    def __init__(self, size=32, channels=3, emb_size=64):
        """
        Conv model used for CIFAR-10 and PlantVillage
        :param size: size of image
        :param channels: number of channels in an image
        :param emb_size: size of output embeddings
        """
        super(SSDL, self).__init__()
        self.conv1 = Conv2D(192, 5, input_shape=(size, size, channels), padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(160, 1, padding='same', activation='relu')
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(96, 1, padding='same', activation='relu')
        self.bn3 = BatchNormalization()
        self.mp1 = MaxPool2D(pool_size=3, strides=2, padding='same', )
        self.conv4 = Conv2D(96, 5, padding='same', activation='relu')
        self.bn4 = BatchNormalization()
        self.conv5 = Conv2D(192, 1, padding='same', activation='relu')
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(192, 1, padding='same', activation='relu')
        self.bn6 = BatchNormalization()
        self.mp2 = MaxPool2D(pool_size=3, strides=2, padding='same', )
        self.conv7 = Conv2D(192, 3, padding='same', activation='relu')
        self.bn7 = BatchNormalization()
        self.conv8 = Conv2D(emb_size, 1, padding='same', activation='linear')
        self.bn8 = BatchNormalization()
        self.ap = GlobalAvgPool2D()

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.mp1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.mp2(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.ap(x)

        return x

