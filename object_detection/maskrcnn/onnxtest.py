import sys
import time
import math
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import pycocotools.mask as mask_util
from PIL import Image

import onnxruntime

# import original modules
sys.path.append('../../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402


CLASSES = [line.rstrip('\n') for line in open('coco_classes.txt')]
IMAGE_PATH = 'demo.jpg'


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize(
        (int(ratio * image.size[0]), int(ratio * image.size[1])),
        Image.BILINEAR
    )

    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    return image


def display_objdetect_image(
        image, boxes, labels, scores, masks, score_threshold=0.7
):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12, 9))
    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(
            mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0: mask_y_1, mask_x_0: mask_x_1
        ]
        im_mask = im_mask[:, :, None]

        # OpenCV version 4.x
        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image = cv2.drawContours(image, contours, -1, 25, 3)

        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )
        ax.annotate(
            CLASSES[label] + ':' + str(np.round(score, 2)),
            (box[0], box[1]),
            color='w',
            fontsize=12
        )
        ax.add_patch(rect)

    ax.imshow(image)
    plt.show()


def test_maskrcnn():
    ss = onnxruntime.InferenceSession("mask_rcnn_R_50_FPN_1x.onnx")
    image = Image.open(IMAGE_PATH)
    input_data = preprocess(image)

    input_name = ss.get_inputs()[0].name
    output_name = ss.get_outputs()[0].name

    out = ss.run([output_name], {input_name: input_data})
    print(len(out))
    print(out[0].shape)


if __name__ == '__main__':
    test_maskrcnn()
