import cv2
import matplotlib
import numpy as np
from PIL import Image


def get_params(arch):
    weight_path = f"{arch}.onnx"
    model_path = f"{arch}.onnx.prototxt"
    return weight_path, model_path


def save(
    pred: np.ndarray,
    output_filename: str,
    original_width: int,
    original_height: int,
    vmin: int | None = 0,
    vmax: int | None = 10,
    cmap: str = "magma_r",
):
    invalid_mask = pred == -99
    mask = np.logical_not(invalid_mask)

    if vmin is None:
        vmin = np.percentile(pred[mask], 2)
    if vmax is None:
        vmax = np.percentile(pred[mask], 85)

    pred = (pred - vmin) / (vmax - vmin)
    pred[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    pred = cmapper(pred, bytes=True)

    img = pred[...]
    img[invalid_mask] = (128, 128, 128, 256)
    img = cv2.resize(img, (original_width, original_height))
    Image.fromarray(img).save(output_filename)
