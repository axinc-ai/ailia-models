import numpy as np
import cv2

from image_utils import normalize_image


def preprocess_frame(
        frame, input_height, input_width, normalize_type='255'
):
    """
    Pre-process the frames taken from the webcam to input to ailia.

    Parameters
    ----------
    frame: numpy array
    input_height: int
        ailia model input height
    input_width: int
        ailia model input width
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': simply dividing by 255.0
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet
        - 'None': no normalization

    Returns
    -------
    img: numpy array
        Image with the propotions of height and width
        adjusted by padding for ailia model input.
    data: numpy array
        Input data for ailia
    """
    height, width = frame.shape[0], frame.shape[1]
    size = np.max((height, width))
    limit = np.min((height, width))
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if size == height:
        img[:, start:fin] = frame    
    else:
        img[start:fin, :] = frame
    
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = cv2.resize(data, (input_height, input_width))
    data = normalize_image(data, normalize_type)
    data = np.expand_dims(np.rollaxis(data, 2, 0), axis=0).astype(np.float32)
    return img, data
