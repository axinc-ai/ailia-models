import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402
import webcamera_utils  # noqa: E402

from yolov4_tiny_utils import post_processing  # noqa: E402


# ======================
# Parameters
# ======================
DETECTION_SIZE_LISTS = ['416','640','1280']

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


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Yolov4-tiny model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-dw', '--detection_width', metavar='DETECTION_WIDTH',
    default='416', choices=DETECTION_SIZE_LISTS,
    help='detection size lists: ' + ' | '.join(DETECTION_SIZE_LISTS)
)
parser.add_argument(
    '-dh', '--detection_height', metavar='DETECTION_HEIGHT',
    default='416', choices=DETECTION_SIZE_LISTS,
    help='detection size lists: ' + ' | '.join(DETECTION_SIZE_LISTS)
)
args = update_parser(parser)

if args.detection_width != "416" or args.detection_height!="416":
    WEIGHT_PATH = 'yolov4-tiny_'+args.detection_width+'_'+args.detection_height+'.onnx'
    MODEL_PATH = 'yolov4-tiny_'+args.detection_width+'_'+args.detection_height+'.onnx.prototxt'
    IMAGE_HEIGHT = int(args.detection_height)
    IMAGE_WIDTH = int(args.detection_width)
else:
    WEIGHT_PATH = 'yolov4-tiny.onnx'
    MODEL_PATH = 'yolov4-tiny.onnx.prototxt'
    IMAGE_HEIGHT = int(args.detection_height)
    IMAGE_WIDTH = int(args.detection_width)

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

    # inference
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
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

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

    # net initialize
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    detector.set_input_shape((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    else:
        # image mode
        recognize_from_image(detector)


if __name__ == '__main__':
    main()
