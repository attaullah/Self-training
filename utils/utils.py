import sys
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def feature_scaling(feats, epsilon=1e-8):
    """
    Scaling features
    :param feats: input features
    :param epsilon: small constant, to avoid division by zero
    :return: scaled features
    """
    feats_norm = np.copy(feats)
    mu = np.zeros((feats.shape[1]))
    sigma = np.zeros((feats.shape[1]))
    for i in range(feats.shape[1]):
        mu[i] = np.mean(feats[:, i])
        sigma[i] = np.std(feats[:, i], ddof=1)
        # print (sigma)
        if sigma[i] == 0:
            sigma[i] = epsilon
        feats_norm[:, i] = (feats[:, i] - float(mu[i])) / float(sigma[i])
    return feats_norm, mu, sigma


def date_diff_in_seconds(dt2, dt1):
    """
        Computes difference in two datetime objects

    """
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds


def dhms_from_seconds(seconds):

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds


def program_duration(dt1, prefix=''):
    """
    Returns string  for program duration: #days #hours #minutes #seconds

    """
    dt2 = datetime.now()
    dtwithoutseconds = dt2.replace(second=0, microsecond=0)
    seconds = date_diff_in_seconds(dt2, dt1)
    abc = dhms_from_seconds(seconds)
    if abc[0] > 0:
        text = " {} days, {} hours, {} minutes, {} seconds".format(abc[0], abc[1], abc[2], abc[3])
    elif abc[1] > 0:
        text = " {} hours, {} minutes, {} seconds".format(abc[1], abc[2], abc[3])
    elif abc[2] > 0:
        text = "  {} minutes, {} seconds".format(abc[2], abc[3])
    else:
        text = "  {} seconds".format(abc[2], abc[3])
    return prefix + text + ' at ' + str(dtwithoutseconds)
