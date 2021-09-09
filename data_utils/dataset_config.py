

def data_details(name):
    """
        Returns dataset values of specific parameters e.g. n_label, selection_percentile, sigma.

    :param name:  dataset name
    :return: n_label, selection_percentile, sigma
    """
    if 'mnist' in name:
        n_label = 100
        selection_percentile = 0.1
        sigma = 1.8
        if 'fashion' in name:
            sigma = 3.2
    elif 'svhn' in name.lower():
        n_label = 1000
        sigma = 2.4
        selection_percentile = 0.05
    elif 'cifar' in name.lower():
        n_label = 4000
        selection_percentile = 0.05
        sigma = 1.2
    elif 'plant' in name:
        n_label = 380
        selection_percentile = 0.02
        sigma = 0.4
    else:
        print("Dataset not available")

    return n_label, selection_percentile, sigma
