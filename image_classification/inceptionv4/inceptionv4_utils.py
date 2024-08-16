import cv2
import numpy as np


def central_crop(image, central_fraction):
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start + bbox_h_size, bbox_w_start:bbox_w_start + bbox_w_size]
    return image

def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x

def get_processed_image(img):
    # Load image and convert from BGR to RGB
    im = np.asarray(img)[:, :, ::-1]
    im = central_crop(im, 0.875)
    im = cv2.resize(im, (299, 299))
    im = preprocess_input(im)
    im = im.reshape(-1, 299, 299, 3)

    return im
