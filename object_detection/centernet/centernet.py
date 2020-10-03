import time
import sys
import argparse

import numpy as np
import cv2

import ailia

from utils import preprocess, postprocess

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402
from detector_utils import load_image  # noqa: E402

# ======================
# Parameters
# ======================

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
THRESHOLD = 0.3  # Threshold for filteing for filtering (from 0.0 to 1.0)
K_VALUE = 40  # K value for topK function
OPSET_LISTS = ['10', '11']

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='CenterNet model'
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
    '-o', '--opset', metavar='OPSET',
    default='10', choices=OPSET_LISTS,
    help='opset lists: ' + ' | '.join(OPSET_LISTS)
)
args = parser.parse_args()

if args.opset == "10":
    WEIGHT_PATH = './ctdet_coco_dlav0_1x.onnx'
    MODEL_PATH = './ctdet_coco_dlav0_1x.onnx.prototxt'
else:
    WEIGHT_PATH = './ctdet_coco_dlav0_1x_opset11.onnx'
    MODEL_PATH = './ctdet_coco_dlav0_1x_opset11.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/centernet/'


# ======================
# Secondaty Functions
# ======================
# def preprocess(img, resize=512, rgb_means=(104, 117, 123), swap=(2, 0, 1)):
#     interp_method = cv2.INTER_LINEAR
#     img = cv2.resize(np.array(img),
#                      (resize, resize),
#                      interpolation=interp_method).astype(np.float32)
#     img -= rgb_means
#     # make channel first
#     img = img.transpose(swap)
#     return img[None, ...]

def to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


BASE = int(np.ceil(pow(len(COCO_CATEGORY), 1. / 3)))
COLORS = [to_color(x, BASE) for x in range(len(COCO_CATEGORY))]


def draw_detection(im, bboxes, scores, cls_inds):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(
            imgcv,
            (box[0], box[1]),
            (box[2], box[3]),
            COLORS[cls_indx],
            thick
        )
        mess = '%s: %.3f' % (COCO_CATEGORY[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, COLORS[cls_indx], thick // 3)
    return imgcv


# ======================
# Main functions
# ======================
def detect_objects(org_img, net):
    centernet_image_size = (512, 512)
    img = preprocess(org_img, centernet_image_size)
    net.predict(img)
    res = net.get_results()
    dets = postprocess(
        [output[0] for output in res],
        (org_img.shape[1], org_img.shape[0]),
        K_VALUE,
        THRESHOLD
    )

    boxes = []
    scores = []
    cls_inds = []

    # font_scale = 0.5
    # font = cv2.FONT_HERSHEY_SIMPLEX

    for det in dets:
        # Make sure bboxes are not out of bounds
        xmin, ymin, xmax, ymax = det[:4].astype(np.int)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(org_img.shape[1], xmax)
        ymax = min(org_img.shape[0], ymax)

        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(det[4])
        cls_inds.append(det[5])

    return boxes, scores, cls_inds


def recognize_from_image(filename, detector):
    # load input image
    img = load_image(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            boxes, scores, cls_inds = detect_objects(img, detector)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        boxes, scores, cls_inds = detect_objects(img, detector)

    try:
        print('\n'.join(
            ['pos:{}, ids:{}, score:{:.3f}'.format(
                '(%.1f,%.1f,%.1f,%.1f)' % (box[0], box[1], box[2], box[3]),
                COCO_CATEGORY[int(obj_cls)],
                score
            ) for box, obj_cls, score in zip(boxes, cls_inds, scores)]
        ))
    except:
        # FIXME: do not use base 'except'
        pass

    # show image
    im2show = draw_detection(img, boxes, scores, cls_inds)

    cv2.imwrite(args.savepath, im2show)
    print('Script finished successfully.')

    # cv2.imshow('demo', im2show)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()


def recognize_from_video(video, detector):
    capture = get_capture(args.video)

    while(True):
        ret, img = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        boxes, scores, cls_inds = detect_objects(img, detector)
        img = draw_detection(img, boxes, scores, cls_inds)
        cv2.imshow('frame', img)

        # press q to end video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, detector)
    else:
        # image mode
        recognize_from_image(args.input, detector)


if __name__ == '__main__':
    main()
