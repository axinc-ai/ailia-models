import sys
import time
import math

import numpy as np
import cv2
from PIL import Image

from transformers import AutoTokenizer

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
# from image_utils import load_image, normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from math_utils import sigmoid
# logger
from logging import getLogger  # noqa

from bert_model import language_backbone
from anchor import anchor_generator

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_BKBN_PATH = "backbone.onnx"
MODEL_BKBN_PATH = "backbone.onnx.prototxt"
WEIGHT_BERT_PATH = "language_backbone.onnx"
MODEL_BERT_PATH = "language_backbone.onnx.prototxt"
WEIGHT_RPN_PATH = "rpn.onnx"
MODEL_RPN_PATH = "rpn.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/glip/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GLIP', IMAGE_PATH, SAVE_IMAGE_PATH
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

# def draw_bbox(img, bboxes):
#     return img


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.reshape(N, -1, C, H, W)
    layer = layer.transpose(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)

    return layer


def box_decode(preds, anchors):
    TO_REMOVE = 1
    widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
    ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

    wx, wy, ww, wh = (10., 10., 5., 5.)
    dx = preds[:, 0::4] / wx
    dy = preds[:, 1::4] / wy
    dw = preds[:, 2::4] / ww
    dh = preds[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = np.clip(dw, None, math.log(1000. / 16))
    dh = np.clip(dh, None, math.log(1000. / 16))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]

    pred_boxes = np.zeros_like(preds)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

    return pred_boxes


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    pass


def post_processing(
        box_regression, centerness, anchors,
        box_cls,
        dot_product_logits):
    sampled_boxes = []
    anchors = [anchors]
    anchors = list(zip(*anchors))

    for idx, (b, c, a) in enumerate(zip(box_regression, centerness, anchors)):
        # print(b)
        # print(b.shape)
        # print(c)
        # print(c.shape)
        # print(a)

        if box_cls is not None:
            o = box_cls[idx]
        if dot_product_logits is not None:
            d = dot_product_logits[idx]

        positive_map = {1: [1, 2, 3], 2: [5], 3: [7, 8]}
        sampled_boxes.append(
            forward_for_single_feature_map(b, c, a, o, d, positive_map)
        )

    boxlists = list(zip(*sampled_boxes))
    boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
    if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
        boxlists = self.select_over_all_levels(boxlists)

    return boxlists


def forward_for_single_feature_map(
        box_regression, centerness, anchors,
        box_cls=None,
        dot_product_logits=None,
        positive_map=None, ):
    N, A, H, W = box_regression.shape
    A = A // 4

    # put in the same format as anchors
    if box_cls is not None:
        C = box_cls.shape[1] // A
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = sigmoid(box_cls)

    # binary dot product focal version
    if dot_product_logits is not None:
        # print('Dot Product.')
        dot_product_logits = sigmoid(dot_product_logits)
        scores = convert_grounding_to_od_logits(
            logits=dot_product_logits, box_cls=box_cls,
            positive_map=positive_map)
        box_cls = scores

    box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
    box_regression = box_regression.reshape(N, -1, 4)

    pre_nms_thresh = 0.05
    candidate_inds = box_cls > pre_nms_thresh
    pre_nms_top_n = np.sum(candidate_inds.reshape(N, -1), axis=1)
    pre_nms_top_n = np.clip(pre_nms_top_n, None, pre_nms_top_n)

    centerness = permute_and_flatten(centerness, N, A, 1, H, W)
    centerness = sigmoid(centerness.reshape(N, -1))

    # multiply the classification scores with centerness scores
    box_cls = box_cls * centerness[:, :, None]

    results = []
    for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
            in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
        per_box_cls = per_box_cls[per_candidate_inds]

        top_k_indices = np.argsort(-per_box_cls)[:per_pre_nms_top_n]

        per_candidate_nonzeros = np.stack(per_candidate_inds.nonzero())
        per_candidate_nonzeros = per_candidate_nonzeros.T
        per_candidate_nonzeros = per_candidate_nonzeros[top_k_indices, :]

        per_box_loc = per_candidate_nonzeros[:, 0]
        per_class = per_candidate_nonzeros[:, 1] + 1

        detections = box_decode(
            per_box_regression[per_box_loc, :].reshape(-1, 4),
            per_anchors.bbox[per_box_loc, :].reshape(-1, 4)
        )
        print("--")
        print(detections)
        print(detections.shape)
        continue

        boxlist = BoxList(detections, per_anchors.size, mode="xyxy")

        labels = per_class
        scores = np.sqrt(per_box_cls)

        boxlist = boxlist.clip_to_image(remove_empty=False)
        boxlist = remove_small_boxes(boxlist, self.min_size)

        results.append(boxlist)

    return results


def convert_grounding_to_od_logits(
        logits, box_cls, positive_map):
    scores = np.zeros((logits.shape[0], logits.shape[1], box_cls.shape[2]))
    # 256 -> 80, average for each class

    # score aggregation method
    for label_j in positive_map:
        scores[:, :, label_j - 1] = np.mean(logits[:, :, positive_map[label_j]], axis=-1)

    return scores


def predict(models, img):
    # img = preprocess(img, shape)

    captions = ['bobble heads on top of the shelf']

    tokenizer = models["tokenizer"]
    max_length = 256

    tokenized = tokenizer.batch_encode_plus(
        captions,
        max_length=max_length,
        padding='max_length',
        return_special_tokens_mask=True,
        return_tensors='pt',
        truncation=True)

    # language embedding
    net = models['language_backbone']
    language_dict_features = language_backbone(
        net,
        tokenized.input_ids.numpy(),
        tokenized.attention_mask.numpy(),
        tokenized.token_type_ids.numpy())

    img = np.load("images.npy")

    net = models['backbone']
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'images': img})

    features = output
    hidden = language_dict_features["hidden"]
    masks = language_dict_features["masks"]
    embedding = language_dict_features['embedded']

    net = models['rpn']
    if not args.onnx:
        output = net.predict(features + [hidden, masks])
    else:
        output = net.run(None, {
            "feat0": features[0], "feat1": features[1], "feat2": features[2],
            "feat3": features[3], "feat4": features[4],
            "hidden": hidden, "masks": masks,
        })

    box_cls = output[:5]
    box_regression = output[5:10]
    centerness = output[10:15]
    dot_product_logits = output[15:]

    image_height, image_width = (800, 1066)
    anchors = anchor_generator((image_height, image_width), features)

    post_processing(box_regression, centerness, anchors, box_cls, dot_product_logits)

    return


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

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(models):
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
    check_and_download_models(WEIGHT_BKBN_PATH, MODEL_BKBN_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_BERT_PATH, MODEL_BERT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_RPN_PATH, MODEL_RPN_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        backbone = ailia.Net(MODEL_BKBN_PATH, WEIGHT_BKBN_PATH, env_id=env_id)
        language_backbone = ailia.Net(MODEL_BERT_PATH, WEIGHT_BERT_PATH, env_id=env_id)
        rpn = ailia.Net(MODEL_RPN_PATH, WEIGHT_RPN_PATH, env_id=env_id)
    else:
        import onnxruntime
        backbone = onnxruntime.InferenceSession(WEIGHT_BKBN_PATH)
        language_backbone = onnxruntime.InferenceSession(WEIGHT_BERT_PATH)
        rpn = onnxruntime.InferenceSession(WEIGHT_RPN_PATH)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    models = dict(
        tokenizer=tokenizer,
        backbone=backbone,
        language_backbone=language_backbone,
        rpn=rpn,
    )

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
