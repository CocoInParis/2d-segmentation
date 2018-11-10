import keras.backend as K


def dice_coefficient(y_true: K.tf.Tensor, y_pred: K.tf.Tensor, smooth: float = 1e-5) -> float:
    """
    Computes the dice coefficient between a ground truth and prediction from the network.

    :rtype: float
    :param y_true: Ground truth tensor retrieved from the neural network
    :param y_pred: Prediction tensor generated from the neural network
    :param smooth: Smoothing factor for the dice coefficient
    :return: Scalar quantity ranging from 0 to 1 (inclusive) where 0 indicates there is no similarity between two set
    and 1 indicates the two sets are identical.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> float:
    """
    Modifies the dice coefficient metric to take the form of a loss function.

    :rtype: float
    :param y_true: Ground truth tensor retrieved from the neural network
    :param y_pred: Prediction tensor generated from the neural network
    :return: Scalar quantity ranging from 0 to 1 (inclusive) where 0 indicates the two sets are identical and 1
    indicates the two sets have no similarity at all.
    """
    return 1 - dice_coef(y_true, y_pred)


dice_coef = dice_coefficient
dice_loss = dice_coefficient_loss
