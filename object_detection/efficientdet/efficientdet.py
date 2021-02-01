import sys, os
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
from nms_utils import bb_intersection_over_union  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402

from efficientdet_utils import *

# ======================
# Parameters
# ======================

WEIGHT_EFFICIENTDET_D0_PATH = 'efficientdet-d0.onnx'
WEIGHT_EFFICIENTDET_D1_PATH = 'efficientdet-d1.onnx'
WEIGHT_EFFICIENTDET_D2_PATH = 'efficientdet-d2.onnx'
WEIGHT_EFFICIENTDET_D3_PATH = 'efficientdet-d3.onnx'
WEIGHT_EFFICIENTDET_D4_PATH = 'efficientdet-d4.onnx'
WEIGHT_EFFICIENTDET_D5_PATH = 'efficientdet-d5.onnx'
WEIGHT_EFFICIENTDET_D6_PATH = 'efficientdet-d6.onnx'
WEIGHT_EFFICIENTDET_D7_PATH = 'efficientdet-d7.onnx'
MODEL_EFFICIENTDET_D0_PATH = 'efficientdet-d0.onnx.prototxt'
MODEL_EFFICIENTDET_D1_PATH = 'efficientdet-d1.onnx.prototxt'
MODEL_EFFICIENTDET_D2_PATH = 'efficientdet-d2.onnx.prototxt'
MODEL_EFFICIENTDET_D3_PATH = 'efficientdet-d3.onnx.prototxt'
MODEL_EFFICIENTDET_D4_PATH = 'efficientdet-d4.onnx.prototxt'
MODEL_EFFICIENTDET_D5_PATH = 'efficientdet-d5.onnx.prototxt'
MODEL_EFFICIENTDET_D6_PATH = 'efficientdet-d6.onnx.prototxt'
MODEL_EFFICIENTDET_D7_PATH = 'efficientdet-d7.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/efficientdet/'

IMAGE_PATH = 'img.png'
SAVE_IMAGE_PATH = 'output.png'

obj_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'EfficientDet model', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model', type=str, default='d0',
    choices=(
        'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7',
    ),
    help='choice model'
)
parser.add_argument(
    '-t', '--threshold', type=float, default=0.2,
    help='threshold'
)
parser.add_argument(
    '-it', '--iou_threshold', type=float, default=0.2,
    help='iou_threshold'
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


def nms(boxes, scores, iou_threshold):
    # remove overwrapped detection
    det = []
    keep = []
    for idx in range(len(boxes)):
        obj = boxes[idx]
        is_keep = True
        for idx2 in range(len(det)):
            if not keep[idx2]:
                continue
            box_a = [det[idx2][0], det[idx2][1], det[idx2][2], det[idx2][3]]
            box_b = [obj[0], obj[1], obj[2], obj[3]]
            iou = bb_intersection_over_union(box_a, box_b)
            if iou >= iou_threshold:
                if scores[idx2] <= scores[idx]:
                    keep[idx2] = False
                else:
                    is_keep = False
        det.append(obj)
        keep.append(is_keep)

    ret = []
    for _, idx in sorted(zip(scores, range(len(boxes))), reverse=True):
        if keep[idx]:
            ret.append(idx)

    return ret


def preprocess(img, input_size=512):
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)

    img = (img / 255 - mean) / std

    img_meta = aspectaware_resize_padding(
        img[..., ::-1], input_size, input_size, means=None)
    img = img_meta[0]
    framed_metas = img_meta[1:]

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img, [framed_metas]


def postprocess(
        imgs,
        anchors, regression, classification,
        threshold, iou_threshold):
    transformed_anchors = bbox_transform(anchors, regression)
    transformed_anchors = clip_boxes(transformed_anchors, imgs)

    scores = np.max(classification, axis=2, keepdims=True)
    scores_over_thresh = (scores > threshold)[:, :, 0]

    out = []
    for i in range(imgs.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

        classification_per = classification[i, scores_over_thresh[i, :], ...].transpose(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if 0 < len(anchors_nms_idx):
            a = classification_per[:, anchors_nms_idx]
            scores_ = np.max(a, axis=0)
            classes_ = np.argmax(a, axis=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            out.append({
                'rois': boxes_,
                'class_ids': classes_,
                'scores': scores_,
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

    return imgs


# ======================
# Main functions
# ======================


def predict(img, net):
    dic_input_size = {
        'd0': 512,
        'd1': 640,
        'd2': 768,
        'd3': 896,
        'd4': 1024,
        'd5': 1280,
        'd6': 1280,
        'd7': 1280,
    }
    input_size = dic_input_size[args.model]

    img, framed_metas = preprocess(img, input_size=input_size)

    if not args.onnx:
        net.set_input_shape(img.shape)
        output = net.predict({'imgs': img})
    else:
        output = net.run(
            ['regression', 'classification', 'anchors'],
            {'imgs': img})
    regression, classification, anchors = output

    threshold = args.threshold
    iou_threshold = args.iou_threshold
    out = postprocess(
        img,
        anchors, regression, classification,
        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    return out


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    print(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pred = predict(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        pred = predict(img, net)

    # plot result
    imgs = display(pred, [img])
    cv2.imwrite(args.savepath, imgs[0])

    print('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        pred = predict(frame, net)

        # plot result
        imgs = display(pred, [frame])
        cv2.imshow('frame', imgs[0])

    capture.release()
    print('Script finished successfully.')


def main():
    dic_model = {
        'd0': (WEIGHT_EFFICIENTDET_D0_PATH, MODEL_EFFICIENTDET_D0_PATH),
        'd1': (WEIGHT_EFFICIENTDET_D1_PATH, MODEL_EFFICIENTDET_D1_PATH),
        'd2': (WEIGHT_EFFICIENTDET_D2_PATH, MODEL_EFFICIENTDET_D2_PATH),
        'd3': (WEIGHT_EFFICIENTDET_D3_PATH, MODEL_EFFICIENTDET_D3_PATH),
        'd4': (WEIGHT_EFFICIENTDET_D4_PATH, MODEL_EFFICIENTDET_D4_PATH),
        'd5': (WEIGHT_EFFICIENTDET_D5_PATH, MODEL_EFFICIENTDET_D5_PATH),
        'd6': (WEIGHT_EFFICIENTDET_D6_PATH, MODEL_EFFICIENTDET_D6_PATH),
        'd7': (WEIGHT_EFFICIENTDET_D7_PATH, MODEL_EFFICIENTDET_D7_PATH),
    }
    weight_path, model_path = dic_model[args.model]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    if not args.onnx:
        env_id = ailia.get_gpu_environment_id()
        net = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    if args.video is not None:
        recognize_from_video(args.video, net)
    else:
        recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
