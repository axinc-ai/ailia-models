import sys
import time
import argparse

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402C
from detector_utils import plot_results, load_image  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'yolov3-tiny.opt.onnx'
MODEL_PATH = 'yolov3-tiny.opt.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-tiny/'

IMAGE_PATH = 'couple.jpg'
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
THRESHOLD = 0.4
IOU = 0.45
DETECTION_WIDTH = 416

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Yolov3 tiny model'
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
parser.add_argument(
    '-dw', '--detection_width',
    default=DETECTION_WIDTH,
    help='The detection width and height for yolo. (default: 416)'
)
args = parser.parse_args()


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_image(args.input)
    print(f'input image shape: {img.shape}')

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id
    )
    if int(args.detection_width) != 416:
        detector.set_input_shape(
            int(args.detection_width), int(args.detection_width)
        )

    # inferece
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            detector.compute(img, THRESHOLD, IOU)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        detector.compute(img, THRESHOLD, IOU)

    # plot result
    res_img = plot_results(detector, img, COCO_CATEGORY)
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id
    )
    if int(args.detection_width) != DETECTION_WIDTH:
        detector.set_input_shape(
            int(args.detection_width), int(args.detection_width)
        )

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        res_img = plot_results(detector, frame, COCO_CATEGORY, False)
        cv2.imshow('frame', res_img)

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
