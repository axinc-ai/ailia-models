import numpy as np

__all__ = [
    'grid_sample',
]


def grid_sampler_unnormalize(coord, side, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (side - 1)
    else:
        return ((coord + 1) * side - 1) / 2


def grid_sampler_compute_source_index(coord, size, align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    return coord


def safe_get_zero(image, n, c, x, y, H, W):
    value = np.zeros(1)
    if 0 <= x < W and 0 <= y < H:
        value = image[n, c, y, x]
    return value


def safe_get_border(image, n, c, x, y, H, W):
    if x < 0:
        x = 0
    elif x >= W:
        x = W - 1
    if y < 0:
        y = 0
    elif y >= H:
        y = H - 1
    return image[n, c, y, x]


def safe_get(image, n, c, x, y, H, W, padding_mode='zeros'):
    if padding_mode == 'border':
        return safe_get_border(image, n, c, x, y, H, W)
    else:
        return safe_get_zero(image, n, c, x, y, H, W)


def grid_sample(image, grid, padding_mode='zeros', align_corners=False):
    '''
         input shape = [N, C, H, W]
         grid_shape  = [N, H, W, 2]
    
         output shape = [N, C, H, W]
    '''
    N, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]

    output_tensor = np.zeros_like(image)
    for n in range(N):
        for w in range(grid_W):
            for h in range(grid_H):
                # get corresponding grid x and y
                y = grid[n, h, w, 1]
                x = grid[n, h, w, 0]

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
                for c in range(C):
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
                            (Ia.T * wa) + (Ib.T * wb) + \
                            (Ic.T * wc) + (Id.T * wd)).T

                    output_tensor[n, c, h, w] = out_ch_val

    return output_tensor
