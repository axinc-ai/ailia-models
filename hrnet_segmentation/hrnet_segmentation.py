import cv2
import time
import os
import urllib.request
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import argparse

import ailia


model_names = ['HRNetV2-W48', 'HRNetV2-W18-Small-v1', 'HRNetV2-W18-Small-v2']
img_path = "test.png"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--arch', '-a', metavar="ARCH", default='HRNetV2-W18-Small-v2',
    choices=model_names,
    help='model architecture:  ' + ' | '.join(model_names) + ' (default: HRNetV2-W18-Small-v2)'
)
args = parser.parse_args()
model_name = args.arch


weight_path = "checkpoints/" + model_name + ".onnx"
model_path = weight_path + ".prototxt"


if not os.path.exists(model_path):
   urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/hrnet_segmentation/" + model_path, model_path)
if not os.path.exists(weight_path):
   urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/hrnet_segmentation/" + weight_path, weight_path)

print("Weight path: " + weight_path)


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
        save_img.save(os.path.join(sv_path, name[i]+'.png'))


# load dataset
# TODO resize should be removed here
img = cv2.resize(cv2.imread(img_path), (1024, 512))
img = np.array([img.transpose(2, 0, 1) / 255])

# net initialize
env_id = ailia.get_gpu_environment_id()
print("Environment mode: {} (-1: CPU, 1: GPU)".format(env_id))

net = ailia.Net(model_path, weight_path, env_id=env_id)

# compute time
for i in range(10):
    start = int(round(time.time() * 1000))
    input_blobs = net.get_input_blob_list()
    net.set_input_blob_data(img, input_blobs[0])
    net.update()
    result = net.get_results()
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

# prediction saving
result = np.asarray(result).reshape(1, 19, 128, 256)
save_pred(result, ".", ["result"])
print('Successfully finished !')
