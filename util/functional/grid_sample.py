# (c) 2021 ax Inc.

import numpy as np


def grid_sample(
        image, grid,
        padding_mode='zeros',
        align_corners=False):
    try:
        import torch
        from torch.nn import functional as F

        output = F.grid_sample(
            torch.from_numpy(image),
            torch.from_numpy(grid),
            padding_mode=padding_mode,
            align_corners=align_corners)
        output = output.numpy()
    except ModuleNotFoundError:
        output = _grid_sample(
            image, grid,
            padding_mode=padding_mode,
            align_corners=align_corners)

    return output


def _grid_sample(
        image, grid,
        padding_mode='zeros',
        align_corners=False):
    '''
         input shape = [N, C, H, W]
         grid_shape  = [N, H, W, 2]

         output shape = [N, C, H, W]
    '''
    N, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]

    output_tensor = np.zeros(
        [N, C, grid_H, grid_W],
        dtype=image.dtype)

    # get corresponding grid x and y
    y = grid[:, :, :, 1]
    x = grid[:, :, :, 0]

    y = y[:, np.newaxis, :, :]
    x = x[:, np.newaxis, :, :]
    y = y.repeat(C, axis=1)
    x = x.repeat(C, axis=1)

    c = np.zeros_like(y) + np.arange(C)[np.newaxis, :, np.newaxis, np.newaxis]
    n = np.zeros_like(y) + np.arange(N)[:, np.newaxis, np.newaxis, np.newaxis]
    c = c.astype(int)
    n = n.astype(int)

    # Unnormalize with align_corners condition
    ix = grid_sampler_compute_source_index(x, W, align_corners)
    iy = grid_sampler_compute_source_index(y, H, align_corners)

    x0 = np.floor(ix)
    x1 = x0 + 1

    y0 = np.floor(iy)
    y1 = y0 + 1

    # Get W matrix before I matrix, as I matrix requires Channel information
    wa = (x1 - ix) * (y1 - iy)
    wb = (x1 - ix) * (iy - y0)
    wc = (ix - x0) * (y1 - iy)
    wd = (ix - x0) * (iy - y0)

    # Get values of the image by provided x0,y0,x1,y1 by channel

    # image, n, c, x, y, H, W
    x0 = x0.astype(int)
    y0 = y0.astype(int)
    x1 = x1.astype(int)
    y1 = y1.astype(int)
    Ia = safe_get(image, n, c, x0, y0, H, W, padding_mode)
    Ib = safe_get(image, n, c, x0, y1, H, W, padding_mode)
    Ic = safe_get(image, n, c, x1, y0, H, W, padding_mode)
    Id = safe_get(image, n, c, x1, y1, H, W, padding_mode)
    out_ch_val = (
            (Ia * wa) + (Ib * wb) +
            (Ic * wc) + (Id * wd))

    output_tensor[:, :, :, :] = out_ch_val

    return output_tensor


def grid_sampler_unnormalize(
        coord, side, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (side - 1)
    else:
        return ((coord + 1) * side - 1) / 2


def grid_sampler_compute_source_index(
        coord, size, align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    return coord


def safe_get_border(
        image,
        n, c, x, y, H, W):
    x = np.clip(x, 0, (W - 1))
    y = np.clip(y, 0, (H - 1))
    value = image[n, c, y, x]

    return value


def safe_get_zero(
        image,
        n, c, x, y, H, W):
    x = x + 1
    y = y + 1
    x = np.clip(x, 0, (W + 1))
    y = np.clip(y, 0, (H + 1))
    value = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
    value = value[n, c, y, x]

    return value


def safe_get(
        image,
        n, c, x, y, H, W,
        padding_mode='zeros'):
    if padding_mode == 'border':
        return safe_get_border(image, n, c, x, y, H, W)
    else:
        return safe_get_zero(image, n, c, x, y, H, W)
