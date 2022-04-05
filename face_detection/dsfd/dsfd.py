import sys
import time
from distutils.util import strtobool

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'detector.onnx'
MODEL_PATH = 'detector.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dsfd/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

THRESHOLD = 0.3
IOU = 0.3

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('DSFD model', IMAGE_PATH, SAVE_IMAGE_PATH)
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
    '--aug',
    default=True, metavar='BOOL', type=strtobool,
    help='with multi-scale augmentation'
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

def bbox_vote(det):
    iou_thres = args.iou

    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= iou_thres)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue

        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[:750, :]

    return dets


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

def preprocess(img, scale):
    if scale != 1:
        img = cv2.resize(
            img, None, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]
    shape = ((w // 32 + 1) * 32, (h // 32 + 1) * 32)
    img = cv2.resize(img, shape)

    img = np.expand_dims(img, axis=0)
    img = np.concatenate((img, img), axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(boxes, labels, scores, num_boxes):
    score_threshold = 0.05

    num_boxes = int(num_boxes)
    boxes = boxes[:num_boxes]
    labels = labels[:num_boxes]
    scores = scores[:num_boxes]

    to_keep = scores > score_threshold
    boxes = boxes[to_keep]
    labels = labels[to_keep].reshape((-1, 1))
    scores = scores[to_keep].reshape((-1, 1))

    boxes = np.concatenate(
        [boxes[:, [1]], boxes[:, [0]], boxes[:, [3]], boxes[:, [2]]],
        axis=1)

    det = np.concatenate([boxes, scores], axis=1)

    return det


def predict(net, img):
    aug_scale = args.aug

    img = img[:, :, ::-1]  # BGR -> RGB

    h, w, _ = img.shape
    max_scale = np.sqrt(
        2000 * 2000 / (h * w))
    max_scale = 3 if max_scale > 3 else max_scale

    scale = max_scale if max_scale < 1 else 1
    xyxy = np.array([w, h, w, h], dtype='float32')

    # feedforward
    x0 = preprocess(img, scale)
    output = net.predict([x0]) if not args.onnx else net.run(None, {'tower_0/images:0': x0})
    boxes, labels, scores, num_boxes = output
    det0 = post_processing(boxes[0], labels[0], scores[0], num_boxes[0])
    det0[:, :4] = det0[:, :4] * xyxy

    # flip
    x1 = preprocess(img[:, ::-1, :], scale)
    output = net.predict([x1]) if not args.onnx else net.run(None, {'tower_0/images:0': x1})
    boxes, labels, scores, num_boxes = output
    det1 = post_processing(boxes[0], labels[0], scores[0], num_boxes[0])
    det1[:, :4] = det1[:, :4] * xyxy
    det1[:, [0, 2]] = w - det1[:, [2, 0]]

    # shrink detecting and shrink only detect big face
    st = 0.5 if max_scale >= 0.75 else 0.5 * max_scale
    x2 = preprocess(img, st)
    output = net.predict([x2]) if not args.onnx else net.run(None, {'tower_0/images:0': x2})
    boxes, labels, scores, num_boxes = output
    det2 = post_processing(boxes[0], labels[0], scores[0], num_boxes[0])
    det2[:, :4] = det2[:, :4] * xyxy
    index = np.where(np.maximum(
        det2[:, 2] - det2[:, 0] + 1, det2[:, 3] - det2[:, 1] + 1) > 30)[0]
    det2 = det2[index, :]

    if aug_scale:
        # enlarge one times
        bt = min(2, max_scale) if max_scale > 1 else (st + max_scale) / 2
        x3 = preprocess(img, bt)
        output = net.predict([x3]) if not args.onnx else net.run(None, {'tower_0/images:0': x3})
        boxes, labels, scores, num_boxes = output
        det3 = post_processing(boxes[0], labels[0], scores[0], num_boxes[0])
        det3[:, :4] = det3[:, :4] * xyxy

        # enlarge small image x times for small face
        if max_scale > 2:
            bt *= 2
            while bt < max_scale:
                x = preprocess(img, bt)
                output = net.predict([x]) if not args.onnx else net.run(None, {'tower_0/images:0': x})
                boxes, labels, scores, num_boxes = output
                det_x = post_processing(boxes[0], labels[0], scores[0], num_boxes[0])
                det_x[:, :4] = det_x[:, :4] * xyxy
                det3 = np.row_stack((det3, det_x))
                bt *= 2

            x = preprocess(img, max_scale)
            output = net.predict([x]) if not args.onnx else net.run(None, {'tower_0/images:0': x})
            boxes, labels, scores, num_boxes = output
            det_x = post_processing(boxes[0], labels[0], scores[0], num_boxes[0])
            det_x[:, :4] = det_x[:, :4] * xyxy
            det3 = np.row_stack((det3, det_x))

        # enlarge only detect small face
        if 1 < bt:
            index = np.where(np.minimum(
                det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) < 100)[0]
            det3 = det3[index, :]
        else:
            index = np.where(np.maximum(
                det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) > 30)[0]
            det3 = det3[index, :]

        det = np.row_stack((det0, det1, det2, det3))
    else:
        det = np.row_stack((det0, det1))

    dets = bbox_vote(det)

    return dets


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
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                dets = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            dets = predict(net, img)

        res_img = draw_bbox(img, dets)

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
        dets = predict(net, frame)

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
        logger.info("This model requires 10GB or more memory.")
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=False)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
