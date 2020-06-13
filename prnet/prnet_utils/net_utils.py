import numpy as np


# get valid vertices in the pos map
FACE_IND = np.loadtxt('uv-data/face_ind.txt').astype(np.int32)
# 2 x 68 get kpt
UV_KPT_IND = np.loadtxt('uv-data/uv_kpt_ind.txt').astype(np.int32)


def get_vertices(pos, size):
    all_vertices = np.reshape(pos, [size ** 2, -1])
    return all_vertices[FACE_IND, :]


def get_colors(image, vertices):
    '''
    Returns:
        colors: the corresponding colors of vertices. 
                shape = (num of points, 3). n is 45128 here.
    '''
    [h, w, _] = image.shape
    vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
    vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    return image[ind[:, 1], ind[:, 0], :]  # n x 3


def get_colors_from_texture(texture, size):
    all_colors = np.reshape(texture, [size ** 2, -1])
    return all_colors[FACE_IND, :]


def generate_uv_coords(resolution_op):
    uv_coords = np.meshgrid(range(resolution_op), range(resolution_op))
    uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
    uv_coords = np.reshape(uv_coords, [resolution_op**2, -1])
    uv_coords = uv_coords[FACE_IND, :]
    uv_coords = np.hstack((
        uv_coords[:, :2],
        np.zeros([uv_coords.shape[0], 1])
    ))
    return uv_coords


def get_landmarks(pos):
    """
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    """
    return pos[UV_KPT_IND[1, :], UV_KPT_IND[0, :], :]
