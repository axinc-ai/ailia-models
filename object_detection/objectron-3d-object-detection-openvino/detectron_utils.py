import numpy as np
import cv2

from objectron.dataset import graphics

__all__ = [
    'draw_kp',
]


def normalize(image_shape, unnormalized_keypoints):
    ''' normalize keypoints to image coordinates '''
    assert len(image_shape) in [2, 3]
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape

    keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)
    return keypoints


def draw_kp(
        img, keypoints, normalized=True, num_keypoints=9, label=None):
    '''
    img: numpy three dimensional array
    keypoints: array like with shape [9,2]
    name: path to save
    '''
    img_copy = img.copy()
    # if image transposed
    if img_copy.shape[0] == 3:
        img_copy = np.transpose(img_copy, (1, 2, 0))
    # expand dim with zeros, needed for drawing function API
    expanded_kp = np.zeros((num_keypoints, 3))
    keypoints = keypoints if normalized else normalize(img_copy.shape, keypoints)
    expanded_kp[:, :2] = keypoints
    graphics.draw_annotation_on_image(img_copy, expanded_kp, [num_keypoints])
    # put class label if given
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_copy, str(label), (10, 180), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_copy
