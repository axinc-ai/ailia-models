import numpy as np
import cv2
from scipy.special import softmax

def crop_and_resize(raw_img,WIDTH,HEIGHT,arch,resize):
    if resize=="padding":
        #add padding
        frame,resized_img = webcamera_utils.adjust_frame_size(raw_img, HEIGHT, WIDTH)
        return resized_img
    elif resize=="crop":
        #crop bottom
        scale_x = (WIDTH / raw_img.shape[1])
        crop_y = raw_img.shape[0] * scale_x - HEIGHT
        crop_y = int(crop_y / scale_x)

        img = raw_img[crop_y:, :, :]  #keep aspect
        if arch=="erfnet":
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
        elif arch=="scnn":
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
        return img
    return None

def preprocess(img,arch):
    if arch=="erfnet":
        #channel first
        mean = [103.939, 116.779, 123.68] #bgr
        std = [1, 1, 1]
        img = np.expand_dims(img, 0)
        img = img - np.array(mean)[np.newaxis, np.newaxis, ...]
        img = img / np.array(std)[np.newaxis, np.newaxis, ...]
        img = np.array(img).transpose(0, 3, 1, 2)
    elif arch=="scnn":
        #channel last
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img[np.newaxis, :, :, :]
        img = x.astype(np.float32)
    return img

def postprocess(output,arch):
    if arch=="erfnet":
        output = softmax(output, axis=1)
    elif arch=="scnn":
        output = output.transpose((0, 3, 1, 2))
    return output
