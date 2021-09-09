import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import accuracy_score


def compute_affinity_matrix(x, sigma):
    """
    Computes W = e^{-sigma* |x_i -x_j|^2 }
    :param x: input x
    :param sigma: hyper-parameter sigma
    :return: affinity matrix
    """
    pairwise_dist = pdist(x, 'sqeuclidean')
    pairwise_dist = pairwise_dist.astype('float32')
    return squareform(np.exp(-sigma * pairwise_dist))


def compute_matrix_s(affinity_matrix):
    """
    Compute matrix S, such that S = D^-1/2 * W * D^-1/2
    :param affinity_matrix: affinity Matrix, W from step-1
    :return:
    """
    row_sum = np.sum(affinity_matrix, axis=1)
    d_sqrt = np.sqrt(row_sum * row_sum[:, np.newaxis])
    return np.divide(affinity_matrix, d_sqrt, where=d_sqrt != 0)


def apply_llgc(labeled_embed, unlabeled_embed, labeled_labels, original_unlabeled_labels, pred_unlabeled_labels,
               alpha=0.99, sigma=.8, n_iter=10):
    """
    Local Learning with Global Consistency (LLGC) https://dennyzhou.github.io/papers/LLGC.pdf

    :param labeled_embed: embeddings of labelled examples
    :param unlabeled_embed: embeddings of unlabeled examples
    :param labeled_labels: labels of labelled examples
    :param original_unlabeled_labels: original labels of unlabeled examples, for accuracy calculation
    :param pred_unlabeled_labels: predicted labels of unlabeled examples using shallow classifier
    :param alpha: hyper-parameter of LLGC for controlling propagation
    :param sigma: hyper-parameter of LLGC
    :param n_iter: number of iterations
    :return: predicted labels , prediction score, and accuracy for unlabeled data
    """
    template = '****** LLGC Initial accuracy {:.4f} for labeled {} ,alpha = {} sigma ={}, dimensions of embeddings = {}'
    size_of_labeled = len(labeled_embed)
    all_embed = np.concatenate([labeled_embed, unlabeled_embed])
    n_classes = len(np.unique(labeled_labels))
    # instead of  initializing zero as labels for unlabeled, we use predicted labels.
    combined_labels = np.concatenate((labeled_labels, pred_unlabeled_labels))
    y = np.eye(n_classes)[combined_labels]

    # step -1 construct affinity matrix W
    w = compute_affinity_matrix(all_embed, sigma)
    w = w.astype('float32')
    dims = all_embed.shape[1]
    del all_embed, unlabeled_embed, labeled_embed

    # step -2 Calculate S
    s = compute_matrix_s(w)
    del w
    # Step -3 Iteration 0 F(t+1) = S.F(t)*alpha + (1-alpha)*Y
    # For iter==0:            F(0)=Y
    f = np.dot(s, y) * alpha + (1 - alpha) * y

    # initial accuracy before applying few iterations
    # labels for unlabeled y_hat=argmax F
    result = np.argmax(f, 1)
    acc = accuracy_score(original_unlabeled_labels, result[size_of_labeled:])
    print(template.format(acc, size_of_labeled, alpha, sigma, dims))

    # Step -3 apply for `n_iter` iterations
    for i in range(n_iter):
        f = np.dot(s, f) * alpha + (1 - alpha) * y

    # Step -4 calculate labels for unlabeled y_hat=argmax F
    pred_lbls = np.argmax(f, 1)

    pred_score = np.max(f, 1)
    acc = accuracy_score(pred_lbls[size_of_labeled:], original_unlabeled_labels)

    return pred_lbls[size_of_labeled:], pred_score[size_of_labeled:], acc


def label_spreading(labeled_embed, unlabeled_embed, labeled_labels, original_unlabeled_labels, pred_unlabeled_labels,
                    alpha=0.99, sigma=.8, n_iter=10, regularized=True, kernel='rbf', neighbors=10):
    """
    Scikit-learn based implementation of LLGC.

    :param labeled_embed: embeddings of labelled examples
    :param unlabeled_embed: embeddings of unlabeled examples
    :param labeled_labels: labels of labelled examples
    :param original_unlabeled_labels: original labels of unlabeled examples, for accuracy calculation
    :param pred_unlabeled_labels: predicted labels of unlabeled examples using shallow classifier
    :param alpha: hyper-parameter of LLGC for controlling propagation
    :param sigma: hyper-parameter of LLGC
    :param n_iter: number of iterations
    :param regularized: label spreading or unregularized label propagation
    :param kernel: for calculation of affinity matrix, default radial basis function or knn
    :param neighbors: parameter for kernel if knn is used
    :return: predicted labels , prediction score, and accuracy for unlabeled data
    """
    size_of_labeled = len(labeled_embed)
    all_embed = np.concatenate([labeled_embed, unlabeled_embed])
    # instead of  initializing zero as labels for unlabeled, we use predicted labels.
    combined_labels = np.concatenate((labeled_labels, pred_unlabeled_labels))

    if regularized:
        template = '* Label Spreading shape= {}, Y_input= {} nl={} alpha = {} sigma={}'
        print(template.format(all_embed.shape, combined_labels.shape, size_of_labeled, alpha, sigma))
        lp_model = LabelSpreading(kernel=kernel, n_neighbors=neighbors, alpha=alpha, gamma=sigma,
                                  max_iter=n_iter)
    else:
        template = '****** Label Propagation shape = {}, Y_input shape = {} labeled = {} alpha = {} sigma = {}'
        print(template.format(all_embed.shape, combined_labels.shape, size_of_labeled, alpha, sigma))
        lp_model = LabelPropagation(kernel=kernel, n_neighbors=neighbors, gamma=sigma, max_iter=n_iter
                                    )
    lp_model.fit(all_embed, combined_labels)
    predicted_labels = lp_model.transduction_
    acc = accuracy_score(original_unlabeled_labels, predicted_labels[size_of_labeled:])
    print('Accuracy  {:.4f} after n_iter {}'.format(acc, n_iter))

    f = lp_model.label_distributions_  # lp_model.predict_proba(dm)
    prob = np.max(f, 1)

    return predicted_labels[size_of_labeled:], prob[size_of_labeled:], acc


