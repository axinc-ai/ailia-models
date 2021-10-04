import numpy as np

__all__ = [
    'anchor_generator',
]


def _generate_anchors(base_size, scales, aspect_ratios):
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


def _make_anchor():
    aspect_ratios = (0.5, 1.0, 2.0)
    anchor_strides = (4, 8, 16, 32, 64)
    anchor_sizes = (32, 64, 128, 256, 512)

    cell_anchors = [
        _generate_anchors(
            stride,
            np.array((size,), dtype=np.float) / stride,
            np.array(aspect_ratios, dtype=np.float),
        )
        for stride, size in zip(anchor_strides, anchor_sizes)
    ]

    return cell_anchors


def _grid_anchors(grid_sizes):
    strides = (4, 8, 16, 32, 64)
    cell_anchors = _make_anchor()

    anchors = []
    for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        shifts_x = np.arange(
            0, grid_width * stride, step=stride, dtype=np.float32
        )
        shifts_y = np.arange(
            0, grid_height * stride, step=stride, dtype=np.float32
        )
        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

        anchors.append(
            (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
        )

    return anchors


def anchor_generator(grid_sizes):
    anchors = [
        _grid_anchors(grid_sizes)
    ]
    return anchors
