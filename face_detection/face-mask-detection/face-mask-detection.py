import sys
import time
import argparse

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C
from detector_utils import plot_results, load_image  # noqa: E402C
from nms_utils import nms_between_categories  # noqa: E402C


# ======================
# Parameters
# ======================

MODEL_LISTS = ['yolov3-tiny', 'yolov3', 'mb2-ssd']

IMAGE_PATH = 'ferry.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416

FACE_CATEGORY = ['unmasked', 'masked']

IOU = 0.45


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='masked face detection model'
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
    '-a', '--arch', metavar='ARCH',
    default='yolov3-tiny', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = parser.parse_args()

if args.arch == "yolov3-tiny":
    WEIGHT_PATH = 'face-mask-detection-yolov3-tiny.opt.obf.onnx'
    MODEL_PATH = 'face-mask-detection-yolov3-tiny.opt.onnx.prototxt'
    RANGE = ailia.NETWORK_IMAGE_RANGE_U_FP32
    ALGORITHM = ailia.DETECTOR_ALGORITHM_YOLOV3
    THRESHOLD = 0.4
elif args.arch == "yolov3":
    WEIGHT_PATH = 'face-mask-detection-yolov3.opt.obf.onnx'
    MODEL_PATH = 'face-mask-detection-yolov3.opt.onnx.prototxt'
    RANGE = ailia.NETWORK_IMAGE_RANGE_U_FP32
    ALGORITHM = ailia.DETECTOR_ALGORITHM_YOLOV3
    THRESHOLD = 0.4
else:
    WEIGHT_PATH = 'face-mask-detection-mb2-ssd-lite.obf.onnx'
    MODEL_PATH = 'face-mask-detection-mb2-ssd-lite.onnx.prototxt'
    RANGE = ailia.NETWORK_IMAGE_RANGE_S_FP32
    ALGORITHM = ailia.DETECTOR_ALGORITHM_SSD
    THRESHOLD = 0.2
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/face-mask-detection/'


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
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=RANGE,
        algorithm=ALGORITHM,
        env_id=env_id
    )

    # inference
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

    # nms
    detections = []
    for idx in range(detector.get_object_count()):
        obj = detector.get_object(idx)
        detections.append(obj)
    detections = nms_between_categories(
        detections,
        img.shape[1],
        img.shape[0],
        categories=[0, 1],
        iou_threshold=IOU
    )

    # plot result
    res_img = plot_results(detections, img, FACE_CATEGORY)
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=RANGE,
        algorithm=ALGORITHM,
        env_id=env_id
    )

    capture = get_capture(args.video)

    if args.savepath != SAVE_IMAGE_PATH:
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(args.savepath, fmt, capture.get(cv2.CAP_PROP_FPS), (IMAGE_WIDTH, IMAGE_HEIGHT))
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, resized_img = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)

        detections = []
        for idx in range(detector.get_object_count()):
            obj = detector.get_object(idx)
            detections.append(obj)
        detections = nms_between_categories(
            detections,
            frame.shape[1],
            frame.shape[0],
            categories=[0, 1],
            iou_threshold=IOU
        )

        res_img = plot_results(detections, resized_img, FACE_CATEGORY, False)
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
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
