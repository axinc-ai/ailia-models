import sys
import time
from itertools import product as product
from math import ceil
from logging import getLogger

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from box_utils import decode
from nms_utils import nms_boxes

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'mb1_120x120.onnx'
MODEL_PATH = 'mb1_120x120.onnx.prototxt'
WEIGHT_DET_PATH = 'FaceBoxesProd.onnx'
MODEL_DET_PATH = 'FaceBoxesProd.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/3ddfa/'

IMAGE_PATH = 'emma.jpg'
SAVE_IMAGE_PATH = 'output.png'

THRESHOLD = 0.4
IOU = 0.45
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1080

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    '3DDFA_V2', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use object detection.'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='object confidence threshold'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='IOU threshold for NMS'
)
parser.add_argument(
    '-m', '--model_type', default='xxx', choices=('xxx', 'XXX'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

class PriorBox(object):
    def __init__(self, image_size=None):
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [
            [
                ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)
            ] for step in self.steps
        ]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in
                                    [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in
                                    [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0, j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, a_min=0, a_max=1)

        return output


# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

def det_preprocess(img):
    h, w, _ = img.shape

    if h > IMAGE_HEIGHT:
        scale = IMAGE_HEIGHT / h
    if w * scale > IMAGE_WIDTH:
        scale *= IMAGE_WIDTH / (w * scale)

    h_s = int(scale * h)
    w_s = int(scale * w)
    img = cv2.resize(img, dsize=(w_s, h_s))

    img = img.astype(np.float32)

    img -= (104, 117, 123)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img, scale


def det_post_processing(loc, height, width, scale):
    priorbox = PriorBox(image_size=(height, width))
    prior_data = priorbox.forward()

    variance = [0.1, 0.2]
    boxes = decode(np.squeeze(loc, axis=0), prior_data, variance)

    scale_bbox = np.array([width, height, width, height])
    boxes = boxes * scale_bbox / scale

    return boxes


def face_detect(models, img):
    img, scale = det_preprocess(img)

    net = models["det"]

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'input': img})
    loc, conf = output

    h_s, w_s = img.shape[2:]
    boxes = det_post_processing(loc, h_s, w_s, scale)
    scores = conf[0][:, 1]

    # ignore low scores
    confidence_threshold = 0.05
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    nms_threshold = 0.3
    dets = np.hstack([boxes, scores[:, np.newaxis]])
    keep = nms_boxes(boxes, scores, nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]

    vis_thres = 0.5
    keep = np.where(dets[:, 4] > vis_thres)[0]
    dets = dets[keep, :]

    return dets


def predict(models, img):
    dets = face_detect(models, img)

    n = len(dets)
    if n == 0:
        logger.info(f'No face detected, exit')
        return

    logger.info(f'Detect {n} faces')


def recognize_from_image(models):
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
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(models, img)

        # res_img = draw_bbox(out)
        res_img = img

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = predict(net, img)

        # plot result
        res_img = draw_bbox(frame, out)

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
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        det_net = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        det_net = onnxruntime.InferenceSession(WEIGHT_DET_PATH, providers=providers)

    models = {
        "net": net,
        "det": det_net,
    }

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
