import sys, os
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from east_utils import *

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'east.onnx'
MODEL_PATH = 'east.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/east/'

IMAGE_PATH = 'img_2.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'EAST: An Efficient and Accurate Scene Text Detector model',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
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


def preprocess(img, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param img: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = img.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img, (ratio_h, ratio_w)


def post_processing(
        score_map, geo_map,
        score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    boxes = nms_locality(boxes.astype(np.float64), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def draw_box(img, boxes):
    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            print('({},{}),({},{}),({},{}),({},{})'.format(
                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
            ))
            cv2.polylines(
                img, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                color=(255, 255, 0), thickness=1)

    return img


# ======================
# Main functions
# ======================


def predict(img, net):
    img, (ratio_h, ratio_w) = preprocess(img)

    # feedforward
    if not args.onnx:
        net.set_input_shape(img.shape)
        output = net.predict({'import/input_images:0': img})
    else:
        img_name = net.get_inputs()[0].name
        pixel_pos_scores_name = net.get_outputs()[0].name
        link_pos_scores_name = net.get_outputs()[1].name
        output = net.run([pixel_pos_scores_name, link_pos_scores_name],
                         {img_name: img})

    score, geometry = output
    boxes = post_processing(score_map=score, geo_map=geometry)

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    return boxes


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    print(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            boxes = predict(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        boxes = predict(img, net)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res_img = draw_box(img, boxes)

    # plot result
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        boxes = predict(frame, net)

        # plot result
        res_img = draw_box(frame, boxes)

        # show
        cv2.imshow('frame', res_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(args.video, net)
    else:
        recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
