import tensorflow as tf
from tensorflow.keras import layers, models
from .model import get_network, get_optimizer
from losses.Triplet import triplet_loss


def self_model(arch, ds_config, weights=False, loss_type="cross-entropy", opt="adam", lr=1e-3, self_clf_wt=1,
               ret_self_model=False, self_supervised_labels=6, dropout_rate=0.2):
    """
    Create a self-supervised based combined model.

    :param arch: network architecture, for example wr-28-2
    :param ds_config: dataset configuration
    :param weights: random weights or ImageNet pretrained weights
    :param loss_type: cross-entropy or triplet
    :param opt: optimizer
    :param lr: learning rate
    :param self_clf_wt: weight of self-supervised loss [0,1]
    :param ret_self_model: whether to return self-supervised model or not
    :param self_supervised_labels: number of classes for self-supervised loss calculation, for our task is 6 as number
    for six different geometric transformations
    :param dropout_rate: parameter for regularization
    :return: return combined and supervised model and self-supervised model if ret_self_model=True
    """
    if weights:
        weight = "imagenet"
    else:
        weight = None
    input_shape = (ds_config.size, ds_config.size, ds_config.channels)
    # get base model
    base_model, es = get_network(arch=arch, input_shape=input_shape, weights=weight)

    supervised_input = layers.Input(shape=input_shape, name='supervised_input')
    self_supervised_input = layers.Input(shape=input_shape, name='self-supervised_input')

    supervised_output = base_model(supervised_input)
    self_supervised_output = base_model(self_supervised_input)

    # add extra layers
    if dropout_rate > 0:
        supervised_output = layers.Dropout(dropout_rate, name='super_dropout')(supervised_output)
        self_supervised_output = layers.Dropout(dropout_rate, name='self_dropout')(self_supervised_output)
    if es > 0:
        supervised_output = layers.Dense(es, name='super_es', activation='relu')(supervised_output)
        self_supervised_output = layers.Dense(es, name='self_es', activation='relu')(self_supervised_output)

    metrics = []
    if loss_type == "cross-entropy":
        metrics = ['acc']
        supervised_output = layers.Dense(ds_config.nc, name='supervised_clf_model',
                                         activation='softmax')(supervised_output)
        supervised_loss = 'sparse_categorical_crossentropy'
    else:
        supervised_output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),   # l2-normalisation
                                          name='supervised_clf_model')(supervised_output)
        supervised_loss = triplet_loss

    self_supervised_output = layers.Dense(self_supervised_labels, name='self_supervised_clf_model',
                                          activation='softmax')(self_supervised_output)
    # models creation
    self_supervised_model = models.Model(inputs=[self_supervised_input], outputs=[self_supervised_output])
    supervised_model = models.Model(inputs=[supervised_input], outputs=[supervised_output])
    combined_model = models.Model(inputs=[self_supervised_input, supervised_input],
                                  outputs=[self_supervised_output, supervised_output])
    print(combined_model.summary())

    optimizer = get_optimizer(opt, lr)
    self_supervised_optimizer = get_optimizer(opt, 1e-4)

    self_supervised_loss = 'categorical_crossentropy'  # loss for self-supervision
    # models compile
    self_supervised_model.compile(optimizer=self_supervised_optimizer, metrics=['acc'], loss=self_supervised_loss)
    supervised_model.compile(optimizer=optimizer, metrics=metrics, loss=supervised_loss)
    combined_model.compile(optimizer=optimizer, loss={'self_supervised_clf_model': self_supervised_loss,
                                                      'supervised_clf_model': supervised_loss},
                           loss_weights={'self_supervised_clf_model': self_clf_wt, 'supervised_clf_model': 1.0},
                           metrics=metrics)
    if ret_self_model:
        return combined_model, supervised_model, self_supervised_model
    return combined_model, supervised_model
