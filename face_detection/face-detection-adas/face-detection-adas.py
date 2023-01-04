import sys
import time
from distutils.util import strtobool

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'face-detection-adas-0001.onnx'
MODEL_PATH = 'face-detection-adas-0001.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-detection-adas/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 672

THRESHOLD = 0.3
IOU = 0.3

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('face-detection-adas', IMAGE_PATH, SAVE_IMAGE_PATH)
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
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def draw_bbox(img, dets):
    conf_thres = args.threshold

    for bbox in dets:
        if bbox[4] > conf_thres:
            cv2.rectangle(
                img, (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])), (255, 0, 0), 7)

    return img


# ======================
# Main functions
# ======================

def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def decode_bbox(im_h, im_w, mbox_loc, mbox_conf, prior_box):
    mbox_loc = mbox_loc[0]
    mbox_conf = mbox_conf[0]
    print("mbox_loc---", mbox_loc)
    print("mbox_loc---", mbox_loc.shape)
    print("mbox_conf---", mbox_conf)
    print("mbox_conf---", mbox_conf.shape)
    print("prior_box---", prior_box)
    print("prior_box---", prior_box.shape)

    score_th = 0.5
    nms_th = 0.5
    num_boxes = len(mbox_loc) // 4

    bbox_list = []
    score_list = []
    for index in range(num_boxes):
        score = mbox_conf[index * 2 + 0]
        if score < score_th:
            continue

        prior_x0 = prior_box[0][index * 4 + 0]
        prior_y0 = prior_box[0][index * 4 + 1]
        prior_x1 = prior_box[0][index * 4 + 2]
        prior_y1 = prior_box[0][index * 4 + 3]
        prior_cx = (prior_x0 + prior_x1) / 2.0
        prior_cy = (prior_y0 + prior_y1) / 2.0
        prior_w = prior_x1 - prior_x0
        prior_h = prior_y1 - prior_y0

        box_cx = mbox_loc[index * 4 + 0]
        box_cy = mbox_loc[index * 4 + 1]
        box_w = mbox_loc[index * 4 + 2]
        box_h = mbox_loc[index * 4 + 3]

        prior_variance = [0.1, 0.1, 0.2, 0.2]
        cx = prior_variance[0] * box_cx * prior_w + prior_cx
        cy = prior_variance[1] * box_cy * prior_h + prior_cy
        w = np.exp((box_w * prior_variance[2])) * prior_w
        h = np.exp((box_h * prior_variance[3])) * prior_h

        bbox_list.append([
            int((cx - (w / 2.0)) * im_w),
            int((cy - (h / 2.0)) * im_h),
            int((cx - (w / 2.0)) * im_w) + int(w * im_w),
            int((cy - (h / 2.0)) * im_h) + int(h * im_h),
        ])
        score_list.append(float(score))

    # nms
    keep_index = cv2.dnn.NMSBoxes(
        bbox_list,
        score_list,
        score_threshold=score_th,
        nms_threshold=nms_th,
        # top_k=200,
    )
    nms_bbox_list = []
    nms_score_list = []
    for index in keep_index:
        nms_bbox_list.append(bbox_list[index])
        nms_score_list.append(score_list[index])

    return nms_bbox_list, nms_score_list


def predict(model_info, img):
    im_h, im_w, _ = img.shape

    net = model_info['net']
    prior_box = model_info['prior_box']

    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'data': img})
    mbox_loc, mbox_conf = output

    bboxes, scores = decode_bbox(im_h, im_w, mbox_loc, mbox_conf, prior_box)

    # print(bboxes)
    # print(scores)

    return (bboxes, scores)


def recognize_from_image(model_info):
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
                dets = predict(model_info, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            dets = predict(model_info, img)

        # Draw bbox and score
        bboxes, scores = dets
        res_img = img
        for bbox, score in zip(bboxes, scores):
            cv2.putText(res_img, '{:.3f}'.format(score), (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (255, 0, 0))

        # res_img = draw_bbox(img, dets)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(model_info):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        dets = predict(model_info, frame)

        # plot result
        res_img = draw_bbox(frame, dets)

        # show
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    prior_box = np.squeeze(np.load('mbox_priorbox.npy'))

    model_info = {
        'net': net,
        'prior_box': prior_box,
    }

    if args.video is not None:
        recognize_from_video(model_info)
    else:
        recognize_from_image(model_info)


if __name__ == '__main__':
    main()
