import numpy as np
import sys
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_clf_model(name='knn', n=1, solver='liblinear', max_iter=200):
    """

    :param name: name of the classifier, one of rf:randomForest, lda: Linear Discriminant Analysis,
    lr: logistic regression, and knn: k-nearest-neighbour
    :param n: number of neighbours for KNN
    :param solver: solver of logistic regression. Can be liblinear, lbfgs
    :param max_iter: Number of maximum iterations for logistic regression
    :return: scikit-learn object of a classifier
    """
    if name == 'rf':
        return RandomForestClassifier(random_state=0)
    elif name == 'lda':
        return LinearDiscriminantAnalysis()
    elif name == 'lr':
        return LogisticRegression(solver=solver, max_iter=max_iter)
    elif name == 'knn':
        return KNeighborsClassifier(n_neighbors=n)
    else:
        print("CLF not implemented")
        sys.exit(0)


def shallow_clf_accuracy(labeled_train_feat, train_labels, test_image_feat, test_labels, name='knn', n=1,
                         solver='liblinear'):
    """

    :param labeled_train_feat: training examples' embeddings
    :param train_labels: labels of training examples
    :param test_image_feat: test examples' embeddings
    :param test_labels: labels of test examples
    :param name: name of classifier, rf, lda, lr, knn
    :param n: number of nearest neighbours for KNN
    :param solver: solver if name of classifier is lr(logistic regression), one of liblinear, lbfgs
    :return: computed accuracy and labels
    """
    true_test_labels = np.array(test_labels)
    clf = get_clf_model(name.lower(), n, solver)
    clf.fit(labeled_train_feat, train_labels)
    pred_labels = clf.predict(test_image_feat)
    accuracy = accuracy_score(true_test_labels, pred_labels)
    return pred_labels, accuracy


