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

import ailia

# import original modules
sys.path.append('../../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'mask_rcnn_R_50_FPN_1x.onnx'
MODEL_PATH = 'mask_rcnn_R_50_FPN_1x.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mask_rcnn/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

CLASSES = [line.rstrip('\n') for line in open('coco_classes.txt')]


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Real-time NN for object instance segmentation by Mask R-CNN'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
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
    image = padded_image[np.newaxis, :, :, :]
    return image

def create_figure():
    fig, ax = plt.subplots(1, figsize=(12, 9), tight_layout=True)
    return fig, ax

def display_objdetect_image(
        fig, ax, image, boxes, labels, scores, masks, score_threshold=0.7, savepath=None
):
    """
    Display or Save result

    Parameters
    ----------
    savepath: str
        When savepath is not None, save output image instead of displaying
    """
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

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
        mask_y_0 = max(y_0 - int_box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - int_box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0: mask_y_1, mask_x_0: mask_x_1
        ]
        im_mask = im_mask[:, :, None]

        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]  #cv2.findContours has changed since OpenCV 3.x, but in OpenCV 4.0 it changes back

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
            CLASSES[int(label)] + ':' + str(np.round(score, 2)),
            (box[0], box[1]),
            color='w',
            fontsize=12
        )
        ax.add_patch(rect)

    if savepath is not None:
        ax.imshow(image)
        fig.savefig(savepath, dpi=150)
    else:
        plt.imshow(image)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    image = Image.open(args.input)
    input_data = preprocess(image)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if env_id != -1 and ailia.get_environment(env_id).props=="LOWPOWER":
        env_id = -1 # This model requires fuge gpu memory so fallback to cpu mode
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape(input_data.shape)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            boxes, labels, scores, masks = net.predict([input_data])
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        boxes, labels, scores, masks = net.predict([input_data])

    # postprocessing
    fig, ax = create_figure()
    display_objdetect_image(
        fig, ax, image, boxes, labels, scores, masks, savepath=args.savepath
    )
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if env_id != -1 and ailia.get_environment(env_id).props=="LOWPOWER":
        env_id = -1 # This model requires fuge gpu memory so fallback to cpu mode
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    fig, ax = create_figure()

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_data = preprocess(frame)
        net.set_input_shape(input_data.shape)

        boxes, labels, scores, masks = net.predict([input_data])

        ax.clear()
        display_objdetect_image(fig, ax, frame, boxes, labels, scores, masks)
        plt.pause(.01)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
