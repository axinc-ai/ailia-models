import os
import cv2
from functools import lru_cache

import numpy as np
from PIL import Image


MASK_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mask.png",
)


@lru_cache(maxsize=None)
def load_fixed_mask(resolution):
    mask_image = cv2.imread(MASK_IMAGE)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = (
        cv2.resize(
            mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4
        )
        / 255.0
    )
    mask_image = mask_image.transpose(2, 0, 1)  # HWC -> CHW
    return mask_image


def preprocess_fixed_mask_image(image):
    size = 256
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image = np.array(
        Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
    )
    image = image.transpose(2, 0, 1)  # HWC -> CHW

    mask_image = load_fixed_mask(size)

    pixel_values = (image / 255.0 - 0.5) / 0.5
    masked_pixel_values = pixel_values * mask_image
    return pixel_values, masked_pixel_values, mask_image[0:1]
