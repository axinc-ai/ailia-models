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

import yolov5_utils  # noqa: E402


# ======================
# Parameters
# ======================

MODEL_LISTS = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
DETECTION_SIZE_LISTS = ['640','1280']

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov5/'

IMAGE_PATH = 'bus.jpg'
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
parser = get_base_parser(
    'Yolov5 model', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='yolov5s', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-dw', '--detection_width', metavar='DETECTION_WIDTH',
    default='640', choices=DETECTION_SIZE_LISTS,
    help='detection size lists: ' + ' | '.join(DETECTION_SIZE_LISTS)
)
parser.add_argument(
    '-dh', '--detection_height', metavar='DETECTION_HEIGHT',
    default='640', choices=DETECTION_SIZE_LISTS,
    help='detection size lists: ' + ' | '.join(DETECTION_SIZE_LISTS)
)
args = update_parser(parser)

if args.detection_width != "640" or args.detection_height!="640":
    WEIGHT_PATH = args.arch+'_'+args.detection_width+'_'+args.detection_height+'.onnx'
    MODEL_PATH = args.arch+'_'+args.detection_width+'_'+args.detection_height+'.onnx.prototxt'
    IMAGE_HEIGHT = int(args.detection_height)
    IMAGE_WIDTH = int(args.detection_width)
else:
    WEIGHT_PATH = args.arch+'.onnx'
    MODEL_PATH = args.arch+'.onnx.prototxt'
    IMAGE_HEIGHT = int(args.detection_height)
    IMAGE_WIDTH = int(args.detection_width)

# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = load_image(args.input)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGRA2BGR)
    print(f'input image shape: {org_img.shape}')

    org_img2, img = webcamera_utils.adjust_frame_size(org_img, IMAGE_HEIGHT, IMAGE_WIDTH)

    #img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2, 0, 1])
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, 0)

    # net initialize
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

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
    detect_object = yolov5_utils.post_processing(img, THRESHOLD, IOU, output)

    # plot result
    res_img = plot_results(detect_object[0], org_img2, COCO_CATEGORY)

    # crop
    pad_w = (org_img2.shape[1]-org_img.shape[1]) // 2
    pad_h = (org_img2.shape[0]-org_img.shape[0]) // 2
    res_img = res_img[pad_h:pad_h+org_img.shape[0],pad_w:pad_w+org_img.shape[1],:]

    # plot result
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = None
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

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
        
        frame2, img = webcamera_utils.adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2, 0, 1])
        img = img.astype(np.float32) / 255
        img = np.expand_dims(img, 0)

        output = detector.predict([img])
        detect_object = yolov5_utils.post_processing(
            img, THRESHOLD, IOU, output
        )
        res_img = plot_results(detect_object[0], frame2, COCO_CATEGORY)

        pad_w = (frame2.shape[1]-frame.shape[1]) // 2
        pad_h = (frame2.shape[0]-frame.shape[0]) // 2
        res_img = res_img[pad_h:pad_h+frame.shape[0],pad_w:pad_w+frame.shape[1],:]

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
