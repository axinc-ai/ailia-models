import cv2
import numpy as np
from PIL import Image


class PostProcess:
    def __init__(self):
        POSTPROCESS_WILL_DENOISE = False
        DATA_IMG_SIZE = 256

        self.denoise = POSTPROCESS_WILL_DENOISE
        self.img_size = DATA_IMG_SIZE

    def __call__(self, source: Image, result: Image):
        # TODO: Refract -> name, resize
        source = np.array(source)
        result = np.array(result)

        height, width = source.shape[:2]
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        laplacian_diff = source.astype(float) - cv2.resize(
            small_source, (width, height)
        ).astype(float)
        result = (
            (cv2.resize(result, (width, height)) + laplacian_diff)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        result = Image.fromarray(result).convert("RGB")
        return result
