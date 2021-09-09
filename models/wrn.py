"""
Wide Residual Network models for Keras.
Reference  - [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
Code adopted from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/wide_resnet.py
"""
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import get_source_inputs
import tensorflow.keras.backend as keras_backend

IM32_CE_WEIGHTS = 'weights/32x32-CE-weights.h5'


def wide_residual_network(depth=28, width=2, dropout_rate=0.0, weights=None, input_tensor=None, input_shape=None):
    """
    Instantiate the Wide Residual Network architecture, optionally loading weights pre-trained on ImageNet32x32,
    ImageNet64x43, ImageNet224x224 using cross-entropy and triplet loss.

    :param depth: number or layers in the DenseNet
    :param width: multiplier to the ResNet width (number of filters)
    :param dropout_rate: dropout rate
    :param weights: one of `None` (random initialization) or 'imagenet32', 'imagenet64', 'imagenet-full',
                    'imagenet-triplet
    :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the models.
    :param input_shape:  shape tuple
    :return A tf.keras models instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or imagenet .')

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4) should be divisible by 6.')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not keras_backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_wide_residual_network(img_input, depth, width, dropout_rate)

    # Ensure that the models takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create models.
    model = Model(inputs, x, name='wide-resnet')

    # load weights
    if weights is not None:
        if (depth == 28) and (width == 2):  # currently weights are provided for only WRN-28-2.
            weights_path = IM32_CE_WEIGHTS
            model.load_weights(weights_path)
    return model


def __conv1_block(input):
    x = Conv2D(16, (3, 3), padding='same')(input)

    channel_axis = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def __conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create
    # convolution2d for this input
    if keras_backend.image_data_format() == 'channels_first':
        if init.shape[1] != 16 * k:
            init = Conv2D(16 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init.shape[-1] != 16 * k:
            init = Conv2D(16 * k, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(16 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(16 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = add([init, x])
    return m


def __conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 32 * k, else
    # create convolution2d for this input
    if keras_backend.image_data_format() == 'channels_first':
        if init.shape[1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init.shape[-1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(32 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(32 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = add([init, x])
    return m


def ___conv4_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if keras_backend.image_data_format() == 'th' else -1  # keras_backend.image_dim_ordering()

    # Check if input number of filters is same as 64 * k, else
    # create convolution2d for this input
    if keras_backend.image_data_format() == 'th':
        if init.shape[1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init.shape[-1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(64 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(64 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = add([init, x])
    return m


def __create_wide_residual_network(img_input, depth=28, width=2, dropout=0.0, pool='avg'):
    """
    Creates a Wide Residual Network with specified parameters
    :param img_input: Input tensor or layer
    :param depth: Depth of the network. Compute N = (n - 4) / 6.
           For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
           For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
           For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param width: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :return:a Keras Model
    """

    N = (depth - 4) // 6

    x = __conv1_block(img_input)
    nb_conv = 4

    for i in range(N):
        x = __conv2_block(x, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = __conv3_block(x, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = ___conv4_block(x, width, dropout)
        nb_conv += 2

    if 'avg' in pool:
        x = GlobalAveragePooling2D()(x)

    return x
