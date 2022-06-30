import numpy as np


def im2col(images, filters, stride=1, pad=0):
    if images.ndim == 2:
        images = images.reshape(1, 1, *images.shape)
    elif images.ndim == 3:
        B, I_h, I_w = images.shape
        images = images.reshape(B, 1, I_h, I_w)
    B, C, I_h, I_w = images.shape

    if isinstance(filters, tuple):
        if len(filters) == 2:
            filters = (1, 1, *filters)
        elif len(filters) == 3:
            M, F_h, F_w = filters
            filters = (M, 1, F_h, F_w)
        _, _, F_h, F_w = filters
    else:
        if filters.ndim == 2:
            filters = filters.reshape(1, 1, *filters.shape)
        elif filters.ndim == 3:
            M, F_h, F_w = filters.shape
            filters = filters.reshape(M, 1, F_h, F_w)
        _, _, F_h, F_w = filters.shape

    if isinstance(stride, tuple):
        stride_ud, stride_lr = stride
    else:
        stride_ud = stride
        stride_lr = stride
    if isinstance(pad, tuple):
        pad_ud, pad_lr = pad
    elif isinstance(pad, int):
        pad_ud = pad
        pad_lr = pad
    elif pad == "same":
        pad_ud = 0.5 * ((I_h - 1) * stride_ud - I_h + F_h)
        pad_lr = 0.5 * ((I_w - 1) * stride_lr - I_w + F_w)
    pad_zero = (0, 0)

    O_h = int((I_h - F_h + 2 * pad_ud) // stride_ud + 1)
    O_w = int((I_w - F_w + 2 * pad_lr) // stride_lr + 1)

    result_pad = (pad_ud, pad_lr)
    pad_ud = int(np.ceil(pad_ud))
    pad_lr = int(np.ceil(pad_lr))
    pad_ud = (pad_ud, pad_ud)
    pad_lr = (pad_lr, pad_lr)
    images = np.pad(
        images, [pad_zero, pad_zero, pad_ud, pad_lr], "constant")

    cols = np.empty((B, C, F_h, F_w, O_h, O_w))
    for h in range(F_h):
        h_lim = h + stride_ud * O_h
        for w in range(F_w):
            w_lim = w + stride_lr * O_w
            cols[:, :, h, w, :, :] = \
                images[:, :, h:h_lim:stride_ud, w:w_lim:stride_lr]

    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(C * F_h * F_w, B * O_h * O_w)

    return cols, (O_h, O_w), result_pad


def col2im(cols, I_shape, O_shape, stride=1, pad=0):
    def get_f_shape(i, o, s, p):
        return int(i + 2 * p - (o - 1) * s)

    if len(I_shape) == 2:
        B = C = 1
        I_h, I_w = I_shape
    elif len(I_shape) == 3:
        C = 1
        B, I_h, I_w = I_shape
    else:
        B, C, I_h, I_w = I_shape
    O_h, O_w = O_shape

    if isinstance(stride, tuple):
        stride_ud, stride_lr = stride
    else:
        stride_ud = stride
        stride_lr = stride
    if isinstance(pad, tuple):
        pad_ud, pad_lr = pad
    elif isinstance(pad, int):
        pad_ud = pad
        pad_lr = pad

    F_h = get_f_shape(I_h, O_h, stride_ud, pad_ud)
    F_w = get_f_shape(I_w, O_w, stride_lr, pad_lr)
    pad_ud = int(np.ceil(pad_ud))
    pad_lr = int(np.ceil(pad_lr))
    cols = cols.reshape(C, F_h, F_w, B, O_h, O_w).transpose(3, 0, 1, 2, 4, 5)
    images = np.zeros((B, C, I_h + 2 * pad_ud + stride_ud - 1, I_w + 2 * pad_lr + stride_lr - 1))

    for h in range(F_h):
        h_lim = h + stride_ud * O_h
        for w in range(F_w):
            w_lim = w + stride_lr * O_w
            images[:, :, h:h_lim:stride_ud, w:w_lim:stride_lr] += cols[:, :, h, w, :, :]

    return images[:, :, pad_ud: I_h + pad_ud, pad_lr: I_w + pad_lr]
