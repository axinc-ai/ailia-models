import os
import sys
import time
import math
import numpy as np

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, write_predictions, load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-tiny/'

IMAGE_PATH = 'input.jpg'
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
parser = get_base_parser('Yolov3 tiny model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: '+str(THRESHOLD)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo. (default: '+str(IOU)+')'
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
    help='Use onnx runtime.'
)
parser.add_argument(
    '--quantize',
    action='store_true',
    help='Use quantized model.'
)
args = update_parser(parser)

if args.onnx or args.quantize:
    import onnxruntime

if args.quantize:
    WEIGHT_PATH = 'yolov3-tiny_int8_per_tensor.opt.onnx'
    MODEL_PATH = 'yolov3-tiny_int8_per_tensor.opt.onnx.prototxt'
else:
    WEIGHT_PATH = 'yolov3-tiny.opt.onnx'
    MODEL_PATH = 'yolov3-tiny.opt.onnx.prototxt'

# ======================
# Quantized model functions
# ======================

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw, c = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh))
    new_image = np.zeros((w, h, 3))
    new_image[(h-nh)//2:(h-nh)//2+nh,(w-nw)//2:(w-nw)//2+nw,0:3] = image[0:nh,0:nw,0:3]
    new_image = new_image[:,:,::-1] # bgr to rgb
    return new_image, nw, nh, (w - nw)//2, (h - nh) //2

def detect_quantized_model(detector, image):
    model_image_size = [args.detection_width, args.detection_height]
    boxed_image, nw, nh, ow, oh = letterbox_image(image, model_image_size)

    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])

    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    feed_f = dict(zip(['input_1', 'image_shape', 'iou_threshold', 'layer.score_threshold'],
                        (image_data, np.array([args.detection_height, args.detection_width],dtype='float32').reshape(1, 2),
                        np.array([args.iou], dtype='float32').reshape(1),
                        np.array([args.threshold], dtype='float32').reshape(1))))
    all_boxes, all_scores, indices = detector.run(None, input_feed=feed_f)

    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices:
        out_classes.append(idx_[1])
        out_scores.append(all_scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(all_boxes[idx_1])

    detections = []
    for i, c in reversed(list(enumerate(out_classes))):
        box = out_boxes[i]
        score = out_scores[i]
        top, left, bottom, right = box
        top = (top - oh) / nh
        left = (left - ow) / nw
        bottom = (bottom - oh) / nh
        right = (right - ow) / nw

        obj = ailia.DetectorObject(
            category=c,
            prob=score,
            x=left,
            y=top,
            w=right - left,
            h=bottom - top)
        detections.append(obj)

    return detections

# ======================
# Main functions
# ======================
def recognize_from_image(detector):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                if args.quantize or args.onnx:
                    detections = detect_quantized_model(detector, img)
                else:
                    detector.compute(img, args.threshold, args.iou)
                    detections = detector
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            if args.quantize or args.onnx:
                detections = detect_quantized_model(detector, img)
            else:
                detector.compute(img, args.threshold, args.iou)
                detections = detector

        # plot result
        res_img = plot_results(detections, img, COCO_CATEGORY)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction:
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, detections, img, COCO_CATEGORY)

    if args.profile:
        print(detector.get_summary())

    logger.info('Script finished successfully.')


def recognize_from_video(detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    if args.write_prediction:
        frame_count = 0
        frame_digit = int(math.log10(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)
        video_name = os.path.splitext(os.path.basename(args.video))[0]

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        if args.quantize or args.onnx:
            detections = detect_quantized_model(detector, img)
        else:
            detector.compute(img, args.threshold, args.iou)
            detections = detector
        res_img = plot_results(detections, frame, COCO_CATEGORY, False)
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_prediction:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, detections, frame, COCO_CATEGORY)
            frame_count += 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.quantize or args.onnx:
        detector = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
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
        if args.detection_width != DETECTION_SIZE or args.detection_height != DETECTION_SIZE:
            detector.set_input_shape(
                args.detection_width, args.detection_height
            )
        if args.profile:
            detector.set_profile_mode(True)

    if args.video is not None:
        # video mode
        recognize_from_video(detector)
    else:
        # image mode
        recognize_from_image(detector)


if __name__ == '__main__':
    main()
