import sys
import time

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'yolov3.opt.onnx'
MODEL_PATH = 'yolov3.opt.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3/'

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
parser = get_base_parser('Yolov3 model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-dw', '--detection_width',
    default=DETECTION_WIDTH,
    help='The detection width and height for yolo. (default: 416)'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_image(args.input)
    print(f'input image shape: {img.shape}')

    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
    )
    if int(args.detection_width) != DETECTION_WIDTH:
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
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id,
    )
    if int(args.detection_width) != DETECTION_WIDTH:
        detector.set_input_shape(
            int(args.detection_width), int(args.detection_width)
        )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        res_img = plot_results(detector, frame, COCO_CATEGORY, False)
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
