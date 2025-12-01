import cv2
import copy
import numpy as np
import ailia

from copy import deepcopy
from typing import Tuple

def show_mask(mask, img):
    global area_img
    color = np.array([255, 144, 30])
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    img = (img * ~mask) + (img * mask) * 0.6 + mask_image * 0.4
    area_img = copy.deepcopy(mask_image)

    return img


def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img, p, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)
    for p in neg_points:
        cv2.drawMarker(
            img, p, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
            markerSize=30, thickness=5)

    return img


def show_box(box, img):
    cv2.rectangle(
        img, box[0], box[1], color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0)

    return img

class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def preprocess( x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors

    img_size = 1024
    pixel_mean: List[float] = [123.675, 116.28, 103.53],
    pixel_std: List[float] = [58.395, 57.12, 57.375],

    pixel_mean = np.array(pixel_mean).reshape(-1, 1, 1)
    pixel_std = np.array(pixel_std).reshape(-1, 1, 1)

    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w

    x = np.pad(x, ((0, 0), (0, 0), (0, padh), (0, padw)), mode='constant').astype(np.float32)
    return x


class SamPredictor:
    def __init__(
        self,
    ) -> None:
        super().__init__()
        #img size 1024

        self.transform = ResizeLongestSide(1024)

    def set_image(
        self,
        net,
        image: np.ndarray,
        onnx_flag,
        image_format: str = "RGB",
    ) -> None:

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)

        
        input_image_torch = np.transpose(input_image, (2, 0, 1))[np.newaxis, :, :, :]

        self.features = self.set_torch_image(net,input_image_torch, image.shape[:2],onnx_flag)

        return self.features

    def set_torch_image(
        self,
        net,
        transformed_image,
        original_image_size,
        onnx_flag
    ) -> None:


        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])

        input_image = preprocess(transformed_image)

        if onnx_flag:
            input_name = net.get_inputs()[0].name
            features = net.run([],{input_name:input_image})[0]
        else:
            features = net.run(input_image)[0]

        return features

