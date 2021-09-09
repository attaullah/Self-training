import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers


class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regulariser=None, **kwargs):
        """
        ArcFace layer implementation. For details, see https://arxiv.org/abs/1801.07698
        Code adapted from # https://github.com/4uiiurz1/keras-arcface
        :param n_classes: number of classes
        :param s: radius of hypersphere
        :param m: angular margin penality
        :param regulariser: regulariser
        :param kwargs: other extra arguments to tf.keras.layer
        """
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regulariser)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W', shape=(input_shape[0][-1], self.n_classes), initializer='glorot_uniform',
                                 trainable=True, regularizer=self.regularizer)

    def call(self, inputs, **kwargs):
        x, y = inputs
        # c = keras_backend.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        weights = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ weights
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(keras_backend.clip(logits, -1.0 + keras_backend.epsilon(), 1.0 - keras_backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return None, self.n_classes
