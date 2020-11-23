import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402C
from detector_utils import plot_results, load_image  # noqa: E402C

from yolov4_tiny_utils import post_processing

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'yolov4-tiny.onnx'
MODEL_PATH = 'yolov4-tiny.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov4-tiny/'

IMAGE_PATH = 'dog.jpg'
SAVE_IMAGE_PATH = 'output.png'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.25
IOU = 0.45
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Yolov4-tiny model'
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
# Main functions
# ======================
def recognize_from_image(detector):
    # prepare input data
    org_img = load_image(args.input)
    print(f'input image shape: {org_img.shape}')

    img = cv2.cvtColor(org_img, cv2.COLOR_BGRA2RGB)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.transpose(img, [2, 0, 1])
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, 0)

    # inferece
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = detector.predict([img])
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        output = detector.predict([img])
    detect_object = post_processing(img, THRESHOLD, IOU, output)

    # plot result
    res_img = plot_results(detect_object[0], org_img, COCO_CATEGORY)

    # plot result
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video(detector):
    capture = get_capture(args.video)

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = np.transpose(img, [2, 0, 1])
        img = img.astype(np.float32) / 255
        img = np.expand_dims(img, 0)

        output = detector.predict([img])
        detect_object = post_processing(
            img, THRESHOLD, IOU, output
        )
        res_img = plot_results(detect_object[0], frame, COCO_CATEGORY)

        cv2.imshow('frame', res_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=env_id
    )
    detector.set_input_shape(
        (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    else:
        # image mode
        recognize_from_image(detector)


if __name__ == '__main__':
    main()
