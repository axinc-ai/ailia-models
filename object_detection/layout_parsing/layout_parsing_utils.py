from typing import List, Optional

import os
import sys
import cv2
import numpy as np


def pdf_to_images(
    pdf_filename: str,
    dpi: int = 200,
    paths_only: bool = True,
    output_folder: Optional[str] = None,
) -> List[str]:
    import pdf2image

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            if not os.path.isdir(output_folder):
                print("Failed to create output folder")
                sys.exit(1)

    if output_folder is not None:
        image_paths = pdf2image.convert_from_path(
            pdf_filename,
            dpi=dpi,
            output_folder=output_folder,
            paths_only=paths_only,
        )
    else:
        image_paths = pdf2image.convert_from_path(
            pdf_filename,
            dpi=dpi,
            paths_only=paths_only,
        )

    return image_paths


def preprocess(img, input_size, swap=(2, 0, 1)):
    """Preprocess image data before YoloX inference."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

