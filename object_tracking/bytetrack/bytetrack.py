import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

from bytetrack_utils import batched_nms
from tracker.byte_tracker import BYTETracker

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'bytetrack_x_mot17.onnx'
MODEL_PATH = 'bytetrack_x_mot17.onnx.prototxt'
WEIGHT_MOT20_PATH = 'bytetrack_x_mot20.onnx'
MODEL_MOT20_PATH = 'bytetrack_x_mot20.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/bytetrack/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ByteTrack', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use object detection.'
)
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
    '-m', '--model_type', default='xxx', choices=('xxx', 'XXX'),
    help='model type'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    r = min(h / im_h, w / im_w)
    oh, ow = int(im_h * r), int(im_w * r)

    resized_img = cv2.resize(
        img,
        (ow, oh),
        interpolation=cv2.INTER_LINEAR,
    )

    img = np.ones((h, w, 3)) * 114.0
    img[: oh, : ow] = resized_img

    img = img[:, :, ::-1]  # BGR -> RGB
    img = normalize_image(img, 'ImageNet')

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, r


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    N, L, _ = prediction.shape
    box_corner = np.zeros((N, L, 4))
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # Get score and class with highest confidence
        class_pred = np.argmax(image_pred[:, 5: 5 + num_classes], 1)
        class_conf = image_pred[:, 5: 5 + num_classes][np.arange(L), class_pred]
        conf_mask = (image_pred[:, 4] * class_conf >= conf_thre)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate(
            (image_pred[:, :5], class_conf.reshape(-1, 1), class_pred.reshape(-1, 1)),
            axis=1)
        detections = detections[conf_mask]
        if len(detections) == 0:
            continue

        nms_out_index = batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        output[i] = detections

    return output


def predict(net, img):
    shape = (800, 1440)
    img, scale = preprocess(img, shape)

    # feedforward
    output = net.predict([img])
    output = output[0]

    confthre = 0.01
    nmsthre = 0.7
    output = postprocess(output, 1, confthre, nmsthre)

    output = output[0]
    output[:, :4] /= scale

    return output


def recognize_from_image(net):
    tracker = BYTETracker()

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
                output = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, img)

        min_box_area = 100

        # run tracking
        online_targets = tracker.update(output)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

        print("online_tlwhs--", len(online_tlwhs))
        print("online_tlwhs--", online_tlwhs[0])
        print("online_ids--", online_ids)
        print("online_ids--", len(online_ids))
        print("online_scores--", online_scores)
        print("online_scores--", len(online_scores))

    # plot result
    # cv2.imwrite(args.savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = predict(net, img)

        # # show
        # cv2.imshow('frame', res_img)
        #
        # # save results
        # if writer is not None:
        #     writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    dic_model = {
        'xxx': (WEIGHT_PATH, MODEL_PATH),
        'XXX': (WEIGHT_MOT20_PATH, MODEL_MOT20_PATH),
    }
    weight_path, model_path = dic_model[args.model_type]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # load model
    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
