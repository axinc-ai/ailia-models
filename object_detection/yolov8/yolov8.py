import os
import sys
import time
import math

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import plot_results, write_predictions, load_image  # noqa
import webcamera_utils  # noqa

# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_YOLOV8N_PATH = 'yolov8n.onnx'
MODEL_YOLOV8N_PATH = 'yolov8n.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov8/'

IMAGE_PATH = 'demo.jpg'
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
DETECTION_SIZE = 416

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Ultralytics YOLOv8', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: ' + str(THRESHOLD) + ')'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo. (default: ' + str(IOU) + ')'
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=DETECTION_SIZE, type=int,
    help='The detection width and height for yolo. (default: 416)'
)
parser.add_argument(
    '-dh', '--detection_height',
    default=DETECTION_SIZE, type=int,
    help='The detection height and height for yolo. (default: 416)'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    size = 640
    r = min(size / im_h, size / im_w)
    oh, ow = int(round(im_h * r)), int(round(im_w * r))
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    dh, dw = size - oh, size - ow
    if 0 and True:
        stride = 32
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(114, 114, 114))  # add border

    img = normalize_image(img, normalize_type='255')

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(preds):
    conf_thres = 0.25
    xc = np.max(preds[:, 4:], axis=1) > conf_thres

    x = preds[0].T[xc[0]]  # confidence


def predict(net, img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'images': img})
    preds = output[0]

    pred = post_processing(preds)

    return preds


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                predict(net, img)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count - 1)} ms')
        else:
            predict(net, img)

        # # plot result
        # res_img = plot_results(detector, img, COCO_CATEGORY)
        # savepath = get_savepath(args.savepath, image_path)
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)
        #
        # # write prediction
        # if args.write_prediction:
        #     pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
        #     write_predictions(pred_file, detector, img, COCO_CATEGORY)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        out = predict(net, frame)

        # # plot result
        # res_img = draw_bbox(frame, out)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    WEIGHT_PATH = WEIGHT_YOLOV8N_PATH
    MODEL_PATH = MODEL_YOLOV8N_PATH

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        cuda = True
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
