# Adapted from https://github.com/Hakuyume/chainer-ssd
import numpy as np


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
        ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes
