import sys
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from .simple import Simple
from .SSDL import SSDL
from .wrn import wide_residual_network
from losses.ArcFace import ArcFace
from losses.Contrastive import contrastive_loss
from losses.Triplet import triplet_loss
import tensorflow_addons as tfa


def adaptive_gpu_memory():
    """
    Helper function for restricting model to occupy only required GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_network(arch, input_shape=(32, 32, 3), weights=None):
    """
    Returns base model
    :param arch: network architecture
    :param input_shape: shapes of input dataset
    :param weights: use pretrained weights or not
    :return: tf.keras model
    """
    adaptive_gpu_memory()
    if 'ssdl' in arch:
        es = -1
        conv_base = SSDL(size=input_shape[0], channels=input_shape[-1])
    elif 'simple' in arch:
        es = -1
        conv_base = Simple(size=input_shape[0], channels=input_shape[-1])
    elif 'vgg16' in arch:
        es = 64
        conv_base = VGG16(input_shape=input_shape, include_top=False, weights=weights, pooling='avg')
    elif 'wrn' in arch:
        es = 64
        dw = arch.split('-')[1:]
        conv_base = wide_residual_network(depth=int(dw[0]), width=int(dw[1]), input_shape=input_shape, weights=weights)
    else:
        print(arch, " : not implemented")
        sys.exit(0)

    return conv_base, es


def get_optimizer(opt, lr):
    """
    Creates an tf.keras.optimizer object
    :param opt: name of optimizer: adam, sgd, and rmsprop
    :param lr: learning rate
    :return: optimizer
    """
    if opt.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=lr)
    elif opt.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=lr)
    elif opt.lower() == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=lr)
    else:
        print('optimizer not implemented')
        sys.exit(1)
    return optimizer


def get_model(arch, data_config,  weights=False, loss_type="cross-entropy", opt="adam", lr=1e-3, dropout_rate=0.2,
              ret_base_model=False):
    """
    Creates a  model compiled on given loss function.
    :param arch: name of network architecture. For example, simple, ssdl, vgg16, and wrn-28-2.
    :param data_config: Details of dataset
    :param weights: use random or ImageNet pretrained weights. By default random
    :param loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param opt: Name of optimizer
    :param lr: learning rate
    :param dropout_rate: dropout rate used for VGG16, and WRn-28-2
    :param ret_base_model: return base model of bigger network models like VGG16 and WRN-28-2
    :return: tf.keras compiled model
    """
    if weights:
        weight = "imagenet"
    else:
        weight = None
    input_shape = (data_config.size, data_config.size, data_config.channels)
    # get base model
    conv_base, es = get_network(arch=arch, input_shape=input_shape, weights=weight)
    # add classification head
    model = models.Sequential()
    model.add(conv_base)
    if es > 0:  # extra layers are added for vgg and wrn
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))
        model.add(layers.Dense(es, activation=None, name="fc1"))

    # set up loss
    metrics = []  # None
    if 'face' in loss_type:
        input1 = Input(shape=input_shape)
        label = Input(shape=(data_config.nc,))
        x = conv_base(input1)
        if es > 0:  # extra layers are added for vgg and wrn
            x = layers.Dropout(dropout_rate, name="dropout_out")(x)
            x = layers.Dense(es, activation=None, name="es")(x)
        output = ArcFace(n_classes=data_config.nc, name='output')([x, label])
        loss = 'categorical_crossentropy'
        model = models.Model([input1, label], output)

    elif 'contrast' in loss_type:
        model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2-normalisation"))  # l2-normalisation
        loss = contrastive_loss

    elif 'triplet' in loss_type:
        model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2-normalisation"))  # l2-normalisation
        loss = triplet_loss

    else:   # default cross-entropy loss
        model.add(layers.Dense(data_config.nc, activation='softmax', name="fc_out"))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['acc']

    learning_rate = lr
    optimizer = get_optimizer(opt, learning_rate)
    # compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.build((None,) + input_shape)
    model.summary()
    if ret_base_model:
        return model, [conv_base, loss, optimizer, metrics]
    return model


def get_callbacks(verb):
    """
    Create tf.keras.callbacks
    :param verb: verbosity of the logs
    :return: callbacks list
    """
    calls = []
    if verb > 2:
        tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
        calls.append(tqdm_callback)

    return calls

