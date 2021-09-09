import os
import numpy as np
from sklearn.metrics import accuracy_score
from utils.utils import feature_scaling
from utils.sallow_classifiers import shallow_clf_accuracy
from utils.llgc import apply_llgc, label_spreading
from scipy.spatial.distance import cdist
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # suppress TF low level logging info
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_utils.FaceGenerator import FaceDataGenerator
from models.model import get_model, get_callbacks
from data_utils import get_dataset

template1 = "Labeled= {} selection={}% iterations= {}"
template2 = 'total selected based on percentile {} having accuracy {:.2f}%'
template3 = 'Length of predicted {},  unlabeled {}'


def set_dataset(dataset, lt, semi=True):
    """
    Create dataset object containing labeled, unlabeled, and test datasets
    :param dataset: Name of dataset. mnist, fashion_mnist, svhn_cropped, cifar10, plant32, plant64 or plant96
    :param lt: One of cross-entropy, triplet, arcface, or contrastive
    :param semi: semi=true means N-labelled, semi=False means all-labelled
    :return: dataset object and dataset details
    """
    one_hot = True if lt.lower() == "arcface" else False
    dso, data_config = get_dataset.read_data_sets(dataset, one_hot, semi)
    return dso, data_config


def set_model(arch, data_config, weights, loss_type="", opt="adam", lr=1e-3):
    """
    setup a model compiled based on given loss function
    :rtype: tf.keras.model
    """
    model = get_model(arch, data_config, weights, loss_type, opt, lr)
    return model


def get_log_name(flags, data_config, prefix=""):
    """
    Creates a name for target directory and file for saving training logs
    :param flags: training flags
    :param data_config: dataset details
    :param prefix: string to be added
    :return: path of directory and name of file for saving logs
    """
    path = prefix + flags.lt + '_logs/' + flags.dataset + '/' + flags.network + '/'

    log_name = str(data_config.n_label) + '-'
    weights = '-w' if flags.weights else ''
    if flags.lt != "cross-entropy":
        log_name += flags.lbl + '-'

    log_name += flags.opt.lower() + weights
    if flags.self_training:
        log_name = log_name + '-self-training-'
        if flags.lt != "cross-entropy":
            log_name += flags.confidence_measure

    return path, log_name


def compute_supervised_accuracy(model, imgs, lbls, ret_labels=False, v=0, bs=100):
    """
    Evaluates accuracy of the model from softmax probabilities.
    :param model: tf.keras.model
    :param imgs: test images
    :param lbls: test labels
    :param ret_labels: return predicted labels or not
    :param v: verbose. By default 0.
    :param bs: size of mini-batch
    :return: accuracy or accuracy and predicted labels
    """
    accuracy = model.evaluate(imgs, lbls, verbose=v, batch_size=bs)
    accuracy = np.round(accuracy[1] * 100., 2)
    if ret_labels:
        pred_lbls = model.predict(imgs, verbose=v, batch_size=bs)
        return accuracy, pred_lbls
    return accuracy


def get_network_embeddings(model, input_images, lt,  bs=100):
    """
    Returns network embeddings, can be used for accuracy calculation for metric learning losses.
    :param model: tf.keras.model
    :param input_images: input images
    :param lt: loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param bs: size of mini-batch
    :return: embeddings
    """
    if 'face' in lt:
        inp = model.input[0]                        # input placeholder
        layer = -3
    else:
        inp = model.input  # input placeholder
        layer = -2
    outputs = model.layers[layer].output       # all layer outputs
    emb = models.Model(inp, outputs)
    feat = emb.predict(input_images, batch_size=bs)

    return feat


def get_network_output(model, input_images, lt="", scaling=False, v=0, bs=100):
    """
    Returns the output of network model on given input images.
    :param model: tf.keras.model
    :param input_images: input images
    :param lt: loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param scaling: perform feature scaling
    :param v: verbosity
    :param bs: size of mini-batch
    :return: output features
    """
    if 'cross-entropy' in lt:
        feat = model.predict(input_images, verbose=v, batch_size=bs)
    else:
        feat = get_network_embeddings(model, input_images, lt)
    if scaling:
        feat, _, _ = feature_scaling(feat)
    return feat


def compute_embeddings_accuracy(model, imgs, lbls, test_imgs, test_lbls, labelling="knn", loss_type="",
                                ret_labels=False, scaling=True):
    """
    Calculate accuracy from embeddings for metric learning losses.
    :param model: tf.keras.model
    :param imgs: labeled images
    :param lbls: labels of labeled images
    :param test_imgs: test images
    :param test_lbls: labels of test images
    :param labelling: type of shallow classifier. can be one of knn: k-nearest-neighbor, lda: linear discriminant
    analysis, rf: random forest, and lr: logistic regression
    :param loss_type: loss_type: One of cross-entropy, triplet, arcface, or contrastive
    :param ret_labels: whether to return predicted labels
    :param scaling: apply feature scaling or not
    :return: test accuracy
    """
    if lbls.ndim > 1:  # for Arcface, convert one-hot encodings to simple
        lbls = np.argmax(lbls, 1)
        test_lbls = np.argmax(test_lbls, 1)
    labels, accuracy = shallow_clf_accuracy(get_network_output(model, imgs, loss_type, scaling=scaling), lbls,
                                            get_network_output(model, test_imgs, loss_type, scaling=scaling),
                                            test_lbls, labelling)
    accuracy = np.round(accuracy * 100., 2)
    if ret_labels:
        return accuracy, labels
    return accuracy


def compute_accuracy(model, train_images, train_labels, test_images, test_labels, loss_type="cross-entropy",
                     labelling="knn"):
    """
    computes test accuracy from either softmax probabilities or from embeddings by training a shallow classifier.
    :rtype: accuracy
    """
    if 'cross-entropy' in loss_type:
        ac = compute_supervised_accuracy(model, test_images, test_labels)
    else:
        ac = compute_embeddings_accuracy(model, train_images, train_labels, test_images, test_labels,
                                         loss_type=loss_type, labelling=labelling)
    return ac


def log_accuracy(model, dso, loss_type="", semi=True, labelling="knn"):
    """
        computes test accuracy from dataset object either by using softmax probabilities or  embeddings by training a
        shallow classifier.

    :rtype: accuracy
    """
    if semi:
        acc = compute_accuracy(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels, dso.test.images,
                               dso.test.labels, loss_type=loss_type, labelling=labelling)
    else:
        acc = compute_accuracy(model, dso.train.images, dso.train.labels, dso.test.images, dso.test.labels,
                               loss_type=loss_type, labelling=labelling)
    return acc


def start_training(model, dso, lt="", epochs=100, semi=True, bs=100):
    """
    Starts training
    :param model: tf.keras.model
    :param dso: dataset object containing train.labeled, train.unlabeled, and test datasets
    :param lt: loss type
    :param epochs: training for the number of epochs
    :param semi: semi=True : N-labelled, semi=False: All-labelled
    :param bs:
    """
    if semi:  # N-labelled
        do_training(model, dso.train.labeled_ds.images, dso.train.labeled_ds.labels, dso.test.images, dso.test.labels,
                    train_iter=epochs, lt=lt, batch_size=bs)
    else:  # all-labelled examples
        do_training(model, dso.train.images, dso.train.labels, dso.test.images, dso.test.labels,
                    train_iter=epochs, lt=lt, batch_size=bs)


def do_training(model, images, labels, test_images, test_labels, train_iter=10, batch_size=100, print_test_error=False,
                verb=1, hflip=False, vf=10, lt=""):
    """
    Start training on given training images.

    :param model: tf.keras.model
    :param images: training images
    :param labels: labels of training images
    :param test_images: test images used for calculation of accuracy
    :param test_labels: labels for test images
    :param train_iter: number of epochs for training
    :param batch_size: size of mini-batch
    :param print_test_error: print test error after each epoch
    :param verb: verbose
    :param hflip: horizontal-flip for augmentations
    :param vf: verbose frequency, show training logs after `vf` epochs
    :param lt: loss type
    :return: training history
    """
    calls = get_callbacks(verb)

    aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, fill_mode='nearest',
                             horizontal_flip=hflip)

    steps_per_epoch = int(len(labels) / batch_size)
    test_aug = ImageDataGenerator()
    test_generator = test_aug.flow(test_images, test_labels, batch_size=batch_size)

    if 'face' in lt:
        train_generator = FaceDataGenerator(images, labels, batch_size=batch_size, augmentations=aug)
        test_generator = FaceDataGenerator(test_images, test_labels, batch_size=batch_size)
    else:
        train_generator = aug.flow(images, labels, batch_size=batch_size)

    if print_test_error:
        history = model.fit(train_generator, epochs=train_iter,  verbose=verb,
                            steps_per_epoch=steps_per_epoch, validation_data=test_generator,
                            validation_freq=vf, callbacks=calls)
    else:
        history = model.fit(train_generator, epochs=train_iter, verbose=verb, steps_per_epoch=steps_per_epoch,
                            callbacks=calls)
    return history


def start_self_learning(model, dso, dc, lt, conf, i, mti, bs, logger):
    """
    Start self-learning
    :param model: tf.keras.model
    :param dso: dataset object containing train.labeled, train.unlabeled, and test datasets
    :param dc: dataset configurations details
    :param lt: loss type
    :param conf: confidence measure for selection of pseudo-labels
    :param i: number of meta-iterations
    :param mti: number of epochs in each meta-iteration
    :param bs: batch size
    :param logger: logger object for printing/saving to file
    """
    self_learning(model, dso, lt, conf, logger,  i, dc.sp, mti, bs)


def pseudo_label_selection(imgs, pred_lbls, scores, orig_lbls, p=0.05):
    """
    Select top p% pseudo-labels based on confidence score. Stratified selection.

    :param imgs: images
    :param pred_lbls: predicted labels
    :param scores: confidence score
    :param orig_lbls: original labels of images
    :param p: selection percentile p%
    :return: pseudo-labelled images and their indices
    """
    to_select = int(len(pred_lbls) * p)
    pseudo_images = []
    pseudo_labels = []
    orig_lbls_selected = []
    number_classes = np.unique(pred_lbls)
    per_class = to_select // len(number_classes)
    args = np.argsort(scores)
    indices = []
    for key in number_classes:  # for all classes
        selected = 0
        for index in args:
            if pred_lbls[index] == key:
                pseudo_images.append(imgs[index])
                pseudo_labels.append(pred_lbls[index])
                indices.append(index)
                orig_lbls_selected.append(orig_lbls[index])
                selected += 1
                if per_class == selected:
                    break
    orig_lbls_selected = np.array(orig_lbls_selected)
    pseudo_labels = np.array(pseudo_labels)
    if orig_lbls_selected.ndim > 1:
        acc = accuracy_score(np.argmax(orig_lbls_selected, 1), pseudo_labels) * 100.
    else:
        acc = accuracy_score(orig_lbls_selected, pseudo_labels) * 100.
    return np.array(pseudo_images), pseudo_labels, indices, acc


def assign_labels(model, mdso, unlabeled_imgs, unlabeled_lbls, lt, confidence="1-nn", alpha=0.99, sigma=0.4):
    """
    compute labels and prediction score. For cross-entropy loss, softmax probabilities are used for label assignment and
    prediction score. For metric learning losses, 1-nearest-neighbour or local learning with global consistency (LLGC)
    is used for label assignment and for prediction score.
    :param model: tf.keras.model
    :param mdso: dataset object
    :param unlabeled_imgs:
    :param unlabeled_lbls:
    :param lt: loss type
    :param confidence: confidence measure. Can be 1-nn or llgc
    :param alpha: hyper-parameter for LLGC
    :param sigma: hyper-parameter for LLGC
    :return: predicted labels, prediction score and accuracy for unlabelled examples
    """
    train_labels = mdso.train.labeled_ds.labels
    nc = len(np.unique(train_labels))
    labeled = len(train_labels)
    if unlabeled_lbls.ndim > 1:  # if labels are one-hot encoded
        train_labels = np.argmax(train_labels, 1)
        unlabeled_lbls = np.argmax(unlabeled_lbls, 1)

    if lt == "cross-entropy":
        test_image_feat = model.predict(unlabeled_imgs)
        pred_lbls = np.argmax(test_image_feat, 1)
        calc_score = np.max(test_image_feat, 1)
        calc_score = calc_score * -1.  # negate probs for same notion as distance
    else:   # for other loss functions
        if confidence == 'llgc':
            # if 'svhn' in FLAGS.d:
            #     llgc_meta = label_spreading
            unlabeled_feats = get_network_output(model, unlabeled_imgs)
            labeled_feats = get_network_output(model, mdso.train.labeled_ds.images)
            pred_lbls, shallow_acc = shallow_clf_accuracy(labeled_feats, train_labels, unlabeled_feats, unlabeled_lbls)

            llgc_lbls, llgc_score, llgc_acc = apply_llgc(labeled_feats, unlabeled_feats, mdso.train.labeled_ds.labels,
                                                         unlabeled_lbls, pred_lbls, alpha, sigma)
            # select score and labels for unlabelled examples only
            pred_lbls = llgc_lbls
            calc_score = llgc_score
            calc_score = calc_score * -1.  # negate probs for same notion as distance
            # print('shallow clf acc', shallow_acc, 'after llgc acc', np.round(llgc_acc * 100, 2))
        else:  # default to 1-NN distance as confidence score
            pred_lbls = []
            calc_score = []
            k = 1
            test_image_feat = get_network_output(model, unlabeled_imgs, lt)
            current_labeled_train_feat = get_network_output(model, mdso.train.labeled_ds.images, lt)
            for j in range(len(test_image_feat)):
                search_feat = np.expand_dims(test_image_feat[j], 0)
                # calculate the sqeuclidean similarity and sort
                dist = cdist(current_labeled_train_feat, search_feat, 'sqeuclidean')
                rank = np.argsort(dist.ravel())
                pred_lbls.append(train_labels[rank[:k]])
                calc_score.append(dist[rank[0]])

    pred_lbls = np.array(pred_lbls)
    pred_lbls = pred_lbls.squeeze()
    pred_acc = accuracy_score(unlabeled_lbls, pred_lbls)*100.
    # print('predicted accuracy {:.2f} %'.format(pred_acc))
    calc_score = np.array(calc_score)
    pred_score = calc_score.squeeze()
    return pred_lbls, pred_score, pred_acc


def self_learning(model, mdso, lt, conf,  logger, num_iterations=25, percentile=0.05, epochs=200, bs=100):
    """
    Apply self-learning

    :param model: tf.keras.model
    :param mdso: dataset object containing labeled, unlabeled, and test datasets
    :param lt: loss type
    :param conf: confidence measure. For metric learning losses, can be one of 1-nn or llgc
    :param logger: for printing/saving logs
    :param num_iterations: number of meta-iterations. Default 25
    :param percentile: selection percentile `p%` of pseudo-labels. Default `5%`
    :param epochs: number of epochs in each meta-iteration
    :param bs: mini-batch size
    :return: images [initially-labeled+pseudo-labelled], labels [initially-labeled+pseudo-labelled]
    """
    # Initial labeled data
    imgs = mdso.train.labeled_ds.images
    lbls = mdso.train.labeled_ds.labels
    # Initial unlabeled data
    unlabeled_imgs = mdso.train.unlabeled_ds.images
    unlabeled_lbls = mdso.train.unlabeled_ds.labels
    if lbls.ndim > 1:
        n_classes = len(np.unique(np.argmax(lbls, 1)))
    else:
        n_classes = len(np.unique(lbls))
    n_label = len(lbls)

    logger.info(template1.format(n_label, 100 * percentile, num_iterations))
    logger.info("i-th meta-iteration, unlabelled accuracy, pseudo-label accuracy,test accuracy")

    for i in range(num_iterations):
        print('=============== Meta-iteration = ', str(i + 1), '/', num_iterations, ' =======================')
        # 1- training
        do_training(model, imgs, lbls, mdso.test.images, mdso.test.labels, epochs, bs, lt=lt)
        # 2- predict labels and confidence score
        pred_lbls, pred_score, unlabeled_acc = assign_labels(model, mdso, unlabeled_imgs, unlabeled_lbls, lt, conf)
        # 3- select top p% pseudo-labels
        pseudo_label_imgs, pseudo_labels, indices_of_selected, pseudo_labels_acc = \
            pseudo_label_selection(unlabeled_imgs, pred_lbls, pred_score, unlabeled_lbls, percentile)
        # 4- merging new labeled for next loop iteration
        imgs = np.concatenate([imgs, pseudo_label_imgs], axis=0)
        if lbls.ndim > 1:  # if one-hot encoded
            pseudo_labels = np.eye(n_classes)[pseudo_labels]
        lbls = np.concatenate([lbls, pseudo_labels], axis=0)
        # 5- remove selected pseudo-labelled data from unlabelled data
        unlabeled_imgs = np.delete(unlabeled_imgs, indices_of_selected, 0)
        unlabeled_lbls = np.delete(unlabeled_lbls, indices_of_selected, 0)

        #####################################################################################
        #  print/save accuracies and other information
        test_acc = compute_accuracy(model, mdso.train.labeled_ds.images, mdso.train.labeled_ds.labels, mdso.test.images,
                                    mdso.test.labels, lt)
        print(template2.format(len(indices_of_selected), pseudo_labels_acc))
        print(template3.format(len(lbls) - n_label, len(unlabeled_lbls)))
        print("Acc: unlabeled: {:.2f} %,  test  {:.2f} %".format(unlabeled_acc, test_acc))
        # ith meta-iteration, unlabelled accuracy, pseudo-label accuracy, test accuracy
        logger.info("{},{:.2f},{:.2f},{:.2f}".format(i + 1, unlabeled_acc, pseudo_labels_acc, test_acc))
        #####################################################################################

    return imgs, lbls



