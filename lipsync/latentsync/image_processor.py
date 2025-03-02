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


class AlignRestore(object):
    def __init__(self):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array(
            [[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]]
        )
        self.face_template = self.face_template * ratio
        self.face_size = (
            int(75 * self.crop_ratio[0]),
            int(100 * self.crop_ratio[1]),
        )
        self.p_bias = None

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(
            input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4
        )
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(
            face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4
        )
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask,
            np.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8
            ),
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(
            inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8)
        )
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(
            inv_mask_center, (blur_size + 1, blur_size + 1), 0
        )
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        if np.max(upsample_img) > 256:
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)

        return upsample_img
