import numpy as np

CUBIC_SIZE = 200
CROPPED_SIZE = 88
ORIGINAL_SIZE = 96

POOL_FACTOR = 2


def discretize(coord, cropped_size):
    '''[-1, 1] -> [0, cropped_size]'''
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size

    return (coord - min_normalized) / scale


def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to original range [-cubic_size/2+refpoint, cubic_size/2 + refpoint]
    '''
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size / 2 + refpoint

    return coord


def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32)

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic


def extract_coord_from_output(output, center=True):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    center: if True, add 0.5, default is true
    return: shape (batch, jointNum, 3)
    '''
    assert (len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T

    xyz_output = max_index.reshape([*output.shape[:-3], 3])

    # Note discrete coord can represents real range [coord, coord+1), see function scattering()
    # So, move coord to range center for better fittness
    if center:
        xyz_output = xyz_output + 0.5

    return xyz_output


def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    # points shape: (n, 3)
    coord = points

    # note, will consider points within range [refpoint-cubic_size/2, refpoint+cubic_size/2] as candidates

    # normalize
    coord = (coord - refpoint) / (cubic_size / 2)  # -> [-1, 1]

    # discretize
    coord = discretize(coord, cropped_size)  # -> [0, cropped_size]
    coord += (original_size / 2 - cropped_size / 2)  # move center to original volume

    # resize around original volume center
    resize_scale = new_size / 100
    if new_size < 100:
        coord = coord * resize_scale + original_size / 2 * (1 - resize_scale)
    elif new_size > 100:
        coord = coord * resize_scale - original_size / 2 * (resize_scale - 1)
    else:
        # new_size = 100 if it is in test mode
        pass

    # rotation
    if angle != 0:
        original_coord = coord.copy()
        original_coord[:, 0] -= original_size / 2
        original_coord[:, 1] -= original_size / 2
        coord[:, 0] = original_coord[:, 0] * np.cos(angle) - original_coord[:, 1] * np.sin(angle)
        coord[:, 1] = original_coord[:, 0] * np.sin(angle) + original_coord[:, 1] * np.cos(angle)
        coord[:, 0] += original_size / 2
        coord[:, 1] += original_size / 2

    # translation
    # Note, if trans = (original_size/2 - cropped_size/2), the following translation will
    # cancel the above translation(after discretion). It will be set it when in test mode.
    coord -= trans

    return coord


def generate_cubic_input(points, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)

    # scattering
    cubic = scattering(coord, cropped_size)

    return cubic


def voxelize(points, refpoint):
    sizes = (CUBIC_SIZE, CROPPED_SIZE, ORIGINAL_SIZE)
    new_size, angle, trans = 100, 0, ORIGINAL_SIZE / 2 - CROPPED_SIZE / 2
    input = generate_cubic_input(points, refpoint, new_size, angle, trans, sizes)

    return input.reshape((1, *input.shape))


def evaluate_keypoints(heatmaps, refpoints):
    coords = extract_coord_from_output(heatmaps)
    coords *= POOL_FACTOR
    keypoints = warp2continuous(coords, refpoints, CUBIC_SIZE, CROPPED_SIZE)

    return keypoints
