import cv2
import numpy as np
from PIL import Image


def transform(image, scaled_size):
    # RescaleT part in original repo
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = scaled_size*h/w, scaled_size
    else:
        new_h, new_w = scaled_size, scaled_size*w/h
    new_h, new_w = int(new_h), int(new_w)
    
    image = cv2.resize(image, (scaled_size, scaled_size))

    # ToTensorLab part in original repo
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image/np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 1] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 2] = (image[:, :, 0]-0.485)/0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
        tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225
    return tmpImg.transpose((2, 0, 1))[np.newaxis, :, :, :]


def load_image(image_path, scaled_size):
    image = cv2.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    if 2 == len(image.shape):
        image = image[:, :, np.newaxis]
    return transform(image, scaled_size), h, w


def norm(pred):
    ma = np.max(pred)
    mi = np.min(pred)
    return (pred - mi) / (ma - mi)


def save_result(pred, savepath, srcimg_shape):
    """
    Parameters
    ----------
    srcimg_shape: (h, w)
    """
    # normalization
    pred = norm(pred)

    img = Image.fromarray(pred * 255).convert('RGB')
    img = img.resize(
        (srcimg_shape[1], srcimg_shape[0]),
        resample=Image.BILINEAR
    )
    img.save(savepath)
    
    
