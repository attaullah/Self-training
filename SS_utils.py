import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import get_callbacks
from data_utils.SS_generator import self_supervised_data_generator, combined_generator
from models.SS_model import self_model
from train_utils import compute_accuracy, assign_labels, pseudo_label_selection

template0 = "Labeled= {}, selection={}% iterations= {}"
template1 = 'total selected based on percentile {} having accuracy {:.2f}%'


def get_model(arch, ds_config, weights, loss_type, opt="adam", lr=1e-3, lambda_u=1, ret_self_model=False):
    """
    Get self-supervised combined model, supervised model only and self-supervised model only.

    :param arch: network architecture
    :param ds_config: dataset configuration
    :param weights: random weights or ImageNet pretrained weights
    :param loss_type: for self-supervised: we have applied cross-entropy or triplet
    :param opt: optimizer
    :param lr: learning rate
    :param lambda_u: [0,1] weight of self-supervised loss
    :param ret_self_model: whether to return self-supervised model or not
    :return: combined model, supervised model only and  if ret_self_model=True self-supervised model only
    """
    combined_model, supervised_model, self_supervised_model = self_model(arch, ds_config, weights=weights,
                                                                         loss_type=loss_type, opt=opt, lr=lr,
                                                                         self_clf_wt=lambda_u,
                                                                         ret_self_model=True)
    if ret_self_model:
        return combined_model, supervised_model, self_supervised_model
    return combined_model, supervised_model


def self_supervised_pretraining(model, images, train_iter=10, batch_size=100, verb=1):
    """
    Perform self-supervised pretraining.

    :param model: tf.keras.model
    :param images: training images
    :param train_iter: number of training epochs
    :param verb: verbosity
    :param batch_size: size of mini-batch
    :return: history of training logs
    """
    steps_per_epoch = len(images) // batch_size

    augmentations = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=False,
                                       fill_mode='reflect')
    # each image will be used to generate 6 transformations
    self_supervised_data = augmentations.flow(images, shuffle=True, batch_size=1)
    train_generator = self_supervised_data_generator(self_supervised_data, batch_size)

    # Pretrain self-supervised model
    history = model.fit(train_generator, epochs=train_iter, verbose=verb, steps_per_epoch=steps_per_epoch)
    return history


def combined_training(model, labeled_images, labels, unlabeled_images, test_images, test_labels, semi=True,
                      train_iter=10, batch_size=100, print_test_error=False, verb=1,  vf=10):
    """
    Perform combined training based on self-supervised and supervised losses jointly.

    :param model: combined model
    :param labeled_images: labelled training images
    :param labels: labels of training images
    :param unlabeled_images: unlabelled images for self-supervision
    :param test_images: test images for test accuracy calculation
    :param test_labels: labels for test images
    :param semi: semi=True means N-labelled, semi=False means All-labelled
    :param train_iter: number of training epochs
    :param print_test_error: print test error after `vf` epochs
    :param verb: verbosity
    :param batch_size: size of mini-batch
    :param vf: verbose frequency
    :return: history of training logs
    """
    calls = get_callbacks(verb)

    hflip = False  # as it is one of the six prediction task for self-supervision task

    if semi:  # if N-labelled, concatenate labelled and unlabelled for self-supervision
        unlabeled_images = np.concatenate([labeled_images, unlabeled_images])
    num_of_images = len(unlabeled_images)
    steps_per_epoch = num_of_images // batch_size

    supervised_aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, fill_mode='reflect',
                                        horizontal_flip=hflip)
    self_supervised_aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=False,
                                             fill_mode='reflect')
    test_aug = ImageDataGenerator()

    # applying augmentations
    supervised_data = supervised_aug.flow(labeled_images, labels, shuffle=True, batch_size=1)
    self_supervised_data = self_supervised_aug.flow(unlabeled_images, shuffle=True, batch_size=1)
    test_generator = test_aug.flow(test_images, test_labels, batch_size=batch_size)
    #
    train_generator = combined_generator(supervised_data, self_supervised_data, batch_size)

    # Fit combined model
    if print_test_error:
        history = model.fit(train_generator, epochs=train_iter,  verbose=verb,
                            steps_per_epoch=steps_per_epoch, validation_data=test_generator,
                            validation_freq=vf, callbacks=calls)
    else:
        history = model.fit(train_generator, epochs=train_iter, verbose=verb, steps_per_epoch=steps_per_epoch,
                            callbacks=calls)
    return history


def start_self_supervised_pretraining(self_supervised_model, dso, semi=True, epochs=100, bs=32):
    """
    Start self-supervised pretraining.

    :param self_supervised_model:
    :param dso: dataset object
    :param semi: dataset configuration details
    :param epochs: number of pretraining epochs
    :param bs: mini-batch size

    """
    if semi:
        all_images = np.concatenate((dso.train.labeled_ds.images, dso.train.unlabeled_ds.images))
        self_supervised_pretraining(self_supervised_model, all_images, epochs, bs)
    else:
        self_supervised_pretraining(self_supervised_model, dso.train.images, epochs, bs)


def start_combined_training(combined_model, dso, epochs, semi=True, bs=32):
    """
    Start combined training.

    :param combined_model:
    :param dso: dataset object
    :param bs: size of mini-batch
    :param epochs: number of training epochs
    :param semi: True = N-labelled , False = All-labelled

    """
    if semi:
        combined_training(combined_model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels,
                          dso.train.unlabeled_ds.images, dso.test.images, dso.test.labels, semi=True, train_iter=epochs,
                          batch_size=bs)
    else:
        combined_training(combined_model, dso.train.images, dso.train.labels,
                          dso.train.images, dso.test.images, dso.test.labels, semi=False, train_iter=epochs,
                          batch_size=bs)


def start_combined_self_learning(combined_model, supervised_model, dso, ds_config, loss_type, logger, num_iterations=25,
                                 network_updates=200, bs=32):
    """
    Start self-training with combined training.

    :param combined_model: combined model
    :param supervised_model: supervised model only
    :param dso: dataset object
    :param ds_config: dataset details
    :param loss_type: loss type for supervised loss: cross-entropy or triplet
    :param logger: for print/save info
    :param num_iterations: number of meta-iterations
    :param network_updates: number of epochs per meta-iteration
    :param bs: size mini-batch
    """
    combined_self_learning(combined_model, supervised_model, dso, ds_config.n_label, loss_type, logger=logger,
                           num_iterations=num_iterations, percentile=ds_config.sp,
                           epochs=network_updates, bs=bs)


def combined_self_learning(model, supervised_model, mdso, labeled, lt, logger, num_iterations=25, percentile=0.05,
                           epochs=200, bs=32):
    """
    Apply self-training using combined training.

    :param model: combined model
    :param supervised_model: supervised model
    :param mdso: dataset object
    :param labeled: number of initially labelled examples
    :param lt: loss type cross-entropy or triplet
    :param logger: for printing/saving logs
    :param num_iterations: number of meta-iterations. Default 25
    :param percentile: selection percentile `p%` of pseudo-labels. Default `5%`
    :param epochs: number of epochs in each meta-iteration
    :param bs: mini-batch size
    :return: images [initially-labeled+pseudo-labelled], labels [initially-labeled+pseudo-labelled] and original_labels
    [initially-labeled+pseudo-labelled]
    """
    # Initial labeled data
    imgs = mdso.train.labeled_ds.images
    lbls = mdso.train.labeled_ds.labels
    # Initial unlabeled data
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels  # only needed for accuracy calculation
    n_label = len(lbls)

    logger.info(template0.format(labeled, 100 * percentile, num_iterations))
    logger.info("i-th meta-iteration, unlabelled accuracy,pseudo-label accuracy,test accuracy")

    for i in range(num_iterations):
        print('=============== Meta-iteration = ', str(i + 1), '/', num_iterations, ' =======================')
        # step-1 training
        combined_training(model, imgs, lbls, unlabeled_imgs, mdso.test.images, mdso.test.labels, train_iter=epochs,
                          batch_size=bs)
        # Step-2 predict labels and score for unlabelled examples
        pred_lbls, pred_score, unlabeled_acc = assign_labels(supervised_model, mdso, unlabeled_imgs, unlabeled_lbls, lt)
        # Step-3 select top p%
        pseudo_label_imgs, pseudo_labels, indices_of_selected, pseudo_labels_acc = \
            pseudo_label_selection(unlabeled_imgs, pred_lbls, pred_score, unlabeled_lbls, percentile)
        # 4- merging new labeled for next loop iteration
        imgs = np.concatenate([imgs, pseudo_label_imgs], axis=0)
        lbls = np.concatenate([lbls, pseudo_labels], axis=0)
        # 5- remove selected pseudo-labelled data from unlabelled data
        unlabeled_imgs = np.delete(unlabeled_imgs, indices_of_selected, 0)
        unlabeled_lbls = np.delete(unlabeled_lbls, indices_of_selected, 0)

        #####################################################################################
        #  print/save accuracies and other information
        test_acc = compute_accuracy(supervised_model, mdso.train.labeled_ds.images, mdso.train.labeled_ds.labels,
                                    mdso.test.images, mdso.test.labels, lt)
        print(template1.format(len(indices_of_selected), pseudo_labels_acc))
        print(template0.format(len(lbls) - n_label, len(unlabeled_lbls)))
        print("Acc: unlabeled: {:.2f} %,  test  {:.2f} %".format(unlabeled_acc, test_acc))
        # ith meta-iteration, unlabelled accuracy, pseudo-label accuracy, test accuracy
        logger.info("{},{:.2f},{:.2f},{:.2f}".format(i + 1, unlabeled_acc, pseudo_labels_acc, test_acc))
        #####################################################################################

    return imgs, lbls

