import functools

import numpy as np

from functional import im2col, col2im


def meshgrid(h, w):
    y = np.arange(0, h).reshape(h, 1, 1).repeat(w, axis=1)
    x = np.arange(0, w).reshape(1, w, 1).repeat(h, axis=0)
    arr = np.concatenate([y, x], axis=-1)

    return arr


def delta_border(h, w):
    """
    :param h: height
    :param w: width
    :return: normalized distance to image border,
     wtith min distance = 0 at border and max dist = 0.5 at image center
    """
    lower_right_corner = np.array([h - 1, w - 1]).reshape(1, 1, 2)
    arr = meshgrid(h, w) / lower_right_corner
    dist_left_up = np.min(arr, axis=-1, keepdims=True)
    dist_right_down = np.min(1 - arr, axis=-1, keepdims=True)

    edge_dist = np.min(np.concatenate([dist_left_up, dist_right_down], axis=-1), axis=-1)

    return edge_dist


def get_weighting(h, w, Ly, Lx):
    clip_min_weight = 0.01
    clip_max_weight = 0.5

    weighting = delta_border(h, w)
    weighting = np.clip(weighting, clip_min_weight, clip_max_weight)
    weighting = weighting.reshape(1, h * w, 1).repeat(Ly * Lx, axis=-1)

    return weighting


def get_fold_unfold(x, kernel_size, stride, uf=1, df=1):
    """
    :param x: img of size (bs, c, h, w)
    :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
    """
    bs, nc, h, w = x.shape

    # number of crops in image
    Ly = (h - kernel_size[0]) // stride[0] + 1
    Lx = (w - kernel_size[1]) // stride[1] + 1

    unfold = functools.partial(im2col, filters=kernel_size, stride=stride)
    if uf == 1 and df == 1:
        fold = functools.partial(
            col2im,
            stride=stride)

        weighting = get_weighting(kernel_size[0], kernel_size[1], Ly, Lx)
        weighting = weighting.reshape((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

    elif uf > 1 and df == 1:
        fold = functools.partial(
            col2im,
            stride=(stride[0] * uf, stride[1] * uf))

        weighting = get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx)
        weighting = weighting.reshape((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

    elif df > 1 and uf == 1:
        fold = functools.partial(
            col2im,
            stride=(stride[0] // df, stride[1] // df))

        weighting = get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx)
        weighting = weighting.reshape((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

    else:
        raise NotImplementedError

    return fold, unfold, weighting
