import numpy as np
import cv2

__all__ = [
    'generate_mask_rect',
    'generate_mask_stroke',
]

"""
https://arxiv.org/abs/1806.03589
"""
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_mask_rect(im_shape, mask_shape, rand=True):
    mask = np.zeros((im_shape[0], im_shape[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shape[0] - mask_shape[0])
        of1 = np.random.randint(0, im_shape[1] - mask_shape[1])
    else:
        of0 = (im_shape[0] - mask_shape[0]) // 2
        of1 = (im_shape[1] - mask_shape[1]) // 2
    mask[of0:of0 + mask_shape[0], of1:of1 + mask_shape[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    return mask


def generate_mask_stroke(im_size, parts=16, maxVertex=24, maxLength=100, maxBrushWidth=24, maxAngle=360):
    h, w = im_size[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w)
    mask = np.minimum(mask, 1.0)
    return mask

def pad(img, ds_factor=32, mode='reflect'):
    h, w = img.shape[:2]

    new_h = ds_factor * ((h - 1) // ds_factor + 1)
    new_w = ds_factor * ((w - 1) // ds_factor + 1)

    pad_h = new_h - h
    pad_w = new_w - w
    if new_h != h or new_w != w:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        img = np.pad(img, pad_width[:img.ndim], mode = mode)
    
    return img

def imnormalize(img, mean, std, to_rgb=True):
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img
