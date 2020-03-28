import sys

import cv2
import numpy as np


def normalize_image(image, normalize_type='255'):
    """
    Normalize image

    Parameters
    ----------
    image: numpy array
        The image you want to normalize 
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': simply dividing by 255.0
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet
        - 'None': no normalization

    Returns
    -------
    normalized_image: numpy array
    """
    if normalize_type == 'None':
        return image
    elif normalize_type == '255':
        return image / 255.0
    elif normalize_type == '127.5':
        return image / 127.5 - 1.0
    elif normalize_type == 'ImageNet':
        print('[FIXME] Not Implemented Error')
        sys.exit(1)


def load_image(
        image_path,
        image_shape,
        rgb=True,
        normalize_type='255',
        gen_input_ailia=False
):
    """
    Loads the image of the given path, performs the necessary preprocessing,
    and returns it.

    Parameters
    ----------
    image_path: string
        The path of image which you want to load.
    image_shape: (int, int)
        Resizes the loaded image to the size required by the model.
    rgb: bool, default=True
        Load as rgb image when True, as gray scale image when False.
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': output range: 0 and 1
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet.
        - 'None': no normalization
    gen_input_ailia: bool, default=False
        If True, convert the image to the form corresponding to the ailia.

    Returns
    -------
    image: numpy array
    """
    # rgb == True --> cv2.IMREAD_COLOR
    # rbg == False --> cv2.IMREAD_GRAYSCALE
    image = cv2.imread(image_path, int(rgb))
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, normalize_type)
    image = cv2.resize(image, image_shape)

    if gen_input_ailia:
        image = image.transpose((2, 0, 1))  # channel first
        image = image[np.newaxis, :, :, :]  # (batch_size, channel, h, w)
    
    return image
