# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np


def generate_anchors(
        stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """

    base_size = stride
    scales = np.array(sizes, dtype=np.float) / stride
    aspect_ratios = np.array(aspect_ratios, dtype=np.float),

    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )

    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


anchor_strides = (8, 16, 32, 64, 128)
anchor_sizes = ((64.0,), (128.0,), (256.0,), (512.0,), (1024.0,))
aspect_ratios = 1.0

cell_anchors = [
    generate_anchors(
        anchor_stride,
        size if isinstance(size, (tuple, list)) else (size,),
        aspect_ratios
    )
    for anchor_stride, size in zip(anchor_strides, anchor_sizes)
]


def grid_anchors(grid_sizes):
    strides = (8, 16, 32, 64, 128)

    anchors = []
    for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        shifts_x = np.arange(
            0, grid_width * stride, step=stride, dtype=np.float32)
        shifts_y = np.arange(
            0, grid_height * stride, step=stride, dtype=np.float32)
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

        anchors.append(
            (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
        )

    return anchors


def get_visibility(anchors, image_height, image_width):
    straddle_thresh = 0
    inds_inside = (
            (anchors[..., 0] >= -straddle_thresh)
            & (anchors[..., 1] >= -straddle_thresh)
            & (anchors[..., 2] < image_width + straddle_thresh)
            & (anchors[..., 3] < image_height + straddle_thresh)
    )
    return inds_inside


from collections import namedtuple

BoxList = namedtuple('BoxList', ['bbox', 'image_size', 'visibility'])


def anchor_generator(image_size, feature_maps):
    grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
    anchors_over_all_feature_maps = grid_anchors(grid_sizes)

    image_height, image_width = image_size

    anchors = []
    for anchors_per_feature_map in anchors_over_all_feature_maps:
        inds_inside = get_visibility(anchors_per_feature_map, image_height, image_width)
        anchors.append(BoxList(anchors_per_feature_map, image_size, inds_inside))

    return anchors
