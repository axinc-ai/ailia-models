import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def convert_label(label, inverse=False):
    ignore_label = -1
    label_mapping = {-1: ignore_label, 0: ignore_label, 
                     1: ignore_label, 2: ignore_label, 
                     3: ignore_label, 4: ignore_label, 
                     5: ignore_label, 6: ignore_label, 
                     7: 0, 8: 1, 9: ignore_label, 
                     10: ignore_label, 11: 2, 12: 3, 
                     13: 4, 14: ignore_label, 15: ignore_label, 
                     16: ignore_label, 17: 5, 18: ignore_label, 
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15, 
                     29: ignore_label, 30: ignore_label, 
                     31: 16, 32: 17, 33: 18}

    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label
    

def save_pred(preds, sv_path, name):
    palette = get_palette(256)
    preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
    for i in range(preds.shape[0]):
        pred = convert_label(preds[i], inverse=True)
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        save_img = save_img.resize((1024, 512))

        save_img.save(os.path.join(sv_path, name[i]+'.png'))
