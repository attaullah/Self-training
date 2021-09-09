import tensorflow as tf
import tensorflow_addons as tfa


def pdist_euclidean(feature, squared=False):
    """
    Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = tf.add(
        tf.reduce_sum(tf.square(feature), axis=[1], keepdims=True),
        tf.reduce_sum(
            tf.square(tf.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * tf.matmul(feature,
                                                    tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.sqrt(
            pairwise_distances_squared + tf.cast(error_mask, tf.float32) * 1e-16)
        # math_ops.to_float(error_mask)

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.multiply(
        pairwise_distances, tf.cast(tf.logical_not(error_mask), tf.float32))
    # math_ops.to_float(math_ops.logical_not(error_mask))
    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data]))
    pairwise_distances = tf.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def square_to_vec(distance_matrix):
    """
     Convert a squared form pdist matrix to vector form.
    :param distance_matrix:
    :return:
    """
    ones = tf.ones_like(distance_matrix)
    mask_a = tf.linalg.band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask

    return tf.boolean_mask(distance_matrix, mask)


def get_contrast_batch_labels(y):
    """
        Make contrast labels by taking all the pairwise in y
    :param y: tensor with shape: (batch_size, )
    :return:  tensor with shape: (batch_size * (batch_size-1) // 2, )
    """
    y_col_vec = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    d_y = pdist_euclidean(y_col_vec)
    d_y = square_to_vec(d_y)
    y_contrasts = tf.cast(d_y == 0, tf.int32)
    return y_contrasts


def contrastive_loss(y, z, margin=1.0, metric='euclidean'):
    """
    Computes contrastive loss.

    :param y: ground truth of shape [bsz].
    :param z: hidden vector of shape [bsz, n_features].
    :param margin: hyper-parameter margin
    :param metric: one of 'euclidean' or  'cosine'
    :return: contrastive loss
    """
    # compute pair-wise distance matrix
    if metric == 'euclidean':
        distance_matrix = pdist_euclidean(z)
    else:  # 'cosine':
        distance_matrix = 1 - tf.matmul(z, z, transpose_a=False, transpose_b=True)
    # convert squareform matrix to vector form
    d_vec = square_to_vec(distance_matrix)
    # make contrastive labels
    y_contrasts = get_contrast_batch_labels(y)
    loss = tfa.losses.contrastive_loss(y_contrasts, d_vec, margin=margin)
    return tf.reduce_mean(loss)
