import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from image_utils import load_image, normalize_image  # noqa: E402C
from math_utils import softmax  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

from this_utils import BBox, anchor_generator, box_decode
from this_utils import remove_small_boxes, boxes_nms, boxes_cat
from this_utils import select_over_all_levels, filter_results

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'xxx.onnx'
MODEL_PATH = 'xxx.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/xxx/'

IMAGE_PATH = 'Cars-1900.mp4'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'XXX', IMAGE_PATH, None
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
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.reshape(N, -1, C, H, W)
    layer = layer.transpose(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


# ======================
# Main functions
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = np.array(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    img = normalize_image(img, normalize_type='ImageNet')

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def rpn_post_processing(anchors, objectness, box_regression):
    pre_nms_top_n = 1000
    min_size = 0
    post_nms_top_n = 300
    nms_thresh = 0.7

    sampled_boxes = []
    anchors = zip(*anchors)
    for a, o, b in zip(anchors, objectness, box_regression):
        N, A, H, W = o.shape

        # put in the same format as anchors
        o = permute_and_flatten(o, N, A, 1, H, W).reshape(N, -1)
        o = sigmoid(o)

        b = permute_and_flatten(b, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(pre_nms_top_n, num_anchors)
        topk_idx = np.argsort(-o, axis=1)[:, :pre_nms_top_n]
        o = o[:, topk_idx[0]]

        batch_idx = np.arange(N)[:, None]
        b = b[batch_idx, topk_idx]

        concat_anchors = np.concatenate(a, axis=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = box_decode(
            b.reshape(-1, 4), concat_anchors.reshape(-1, 4),
            weights=(1.0, 1.0, 1.0, 1.0)
        )
        proposals = proposals.reshape(N, -1, 4)

        result = []
        for proposal, score in zip(proposals, o):
            boxes = BBox(bbox=proposal, scores=score)
            boxes = remove_small_boxes(boxes, min_size)
            boxes = boxes_nms(
                boxes,
                nms_thresh,
                max_proposals=post_nms_top_n,
            )
            print("boxlist----", boxes.bbox.shape)
            result.append(boxes)

        sampled_boxes.append(result)

    boxlists = zip(*sampled_boxes)
    boxlist = [boxes_cat(boxlist) for boxlist in boxlists]

    boxlist = select_over_all_levels(boxlist)
    print("boxlist--", boxlist[0].bbox)
    print("boxlist--", boxlist[0].bbox.shape)

    return boxlist[0]


def post_processing(class_logits, box_regression, bbox, ids=None, labels=None):
    prob = softmax(class_logits, -1)

    proposals = box_decode(
        box_regression, bbox,
        weights=(10.0, 10.0, 5.0, 5.0)
    )

    num_classes = prob.shape[1]

    # deafult id is -1
    ids = ids if ids else np.zeros(len(bbox)) - 1

    labels = []

    # this only happens for tracks
    if labels is not None and 0 < len(labels):
        # tracks
        track_inds = np.nonzero(ids >= 0)[0]

        # avoid track bbs be suppressed during nms
        if track_inds:
            prob_cp = np.array(prob)
            prob[track_inds, :] = 0.
            prob[track_inds, labels] = prob_cp[track_inds, labels] + 1.

    boxes = BBox(
        bbox=proposals.reshape(-1, 4),
        scores=prob.reshape(-1),
        ids=ids
    )
    boxes.bbox[:, 0] = boxes.bbox[:, 0].clip(0, max=800 - 1)
    boxes.bbox[:, 1] = boxes.bbox[:, 1].clip(0, max=1280 - 1)
    boxes.bbox[:, 2] = boxes.bbox[:, 2].clip(0, max=800 - 1)
    boxes.bbox[:, 3] = boxes.bbox[:, 3].clip(0, max=1280 - 1)

    boxes = filter_results(boxes, num_classes)

    return boxes


def solver(boxes):
    """
    The solver is to merge predictions from detection branch as well as from track branch.
    The goal is to assign an unique track id to bounding boxes that are deemed tracked
    :param detection: it includes three set of distinctive prediction:
    prediction propagated from active tracks, (2 >= score > 1, id >= 0),
    prediction propagated from dormant tracks, (2 >= score > 1, id >= 0),
    prediction from detection (1 > score > 0, id = -1).
    :return:
    """
    print(boxes.bbox)
    print(boxes.bbox.shape)
    1 / 0

    if len(detection) == 0:
        return [detection]

    track_pool = self.track_pool

    all_ids = detection.get_field('ids')
    all_scores = detection.get_field('scores')
    active_ids = track_pool.get_active_ids()
    dormant_ids = track_pool.get_dormant_ids()
    device = all_ids.device

    active_mask = torch.tensor([int(x) in active_ids for x in all_ids], device=device)

    # differentiate active tracks from dormant tracks with scores
    # active tracks, (3 >= score > 2, id >= 0),
    # dormant tracks, (2 >= score > 1, id >= 0),
    # By doing this, dormant tracks will be merged to active tracks during nms,
    # if they highly overlap
    all_scores[active_mask] += 1.

    nms_detection, nms_ids, nms_scores = self.get_nms_boxes(detection)

    combined_detection = nms_detection
    _ids = combined_detection.get_field('ids')
    _scores = combined_detection.get_field('scores')

    # start track ids
    start_idxs = ((_ids < 0) & (_scores >= self.start_thresh)).nonzero()

    # inactive track ids
    inactive_idxs = ((_ids >= 0) & (_scores < self.track_thresh))
    nms_track_ids = set(_ids[_ids >= 0].tolist())
    all_track_ids = set(all_ids[all_ids >= 0].tolist())
    # active tracks that are removed by nms
    nms_removed_ids = all_track_ids - nms_track_ids
    inactive_ids = set(_ids[inactive_idxs].tolist()) | nms_removed_ids

    # resume dormant mask, if needed
    dormant_mask = torch.tensor([int(x) in dormant_ids for x in _ids], device=device)
    resume_ids = _ids[dormant_mask & (_scores >= self.resume_track_thresh)]
    for _id in resume_ids.tolist():
        track_pool.resume_track(_id)

    for _idx in start_idxs:
        _ids[_idx] = track_pool.start_track()

    active_ids = track_pool.get_active_ids()
    for _id in inactive_ids:
        if _id in active_ids:
            track_pool.suspend_track(_id)

    # make sure that the ids for inactive tracks in current frame are meaningless (< 0)
    _ids[inactive_idxs] = -1

    track_pool.expire_tracks()
    track_pool.increment_frame()

    return [combined_detection]


def predict(rpn, box, tracker, img, anchors, cache={}):
    # shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    # img = preprocess(img, shape)

    # feedforward
    print("1-----------")
    if not args.onnx:
        output = rpn.predict([img])
    else:
        output = rpn.run(None, {'img': img})
    print("2-----------")

    features = output[:5]
    objectness = output[5:10]
    rpn_box_regression = output[10:]
    # print("features--", features[0].shape)
    # print("objectness--", objectness[0].shape)
    # print("rpn_box_regression--", rpn_box_regression[0].shape)
    boxes = rpn_post_processing(anchors, objectness, rpn_box_regression)
    print("3-----------")

    proposals = boxes.bbox

    # roi_heads
    inputs = features[:4] + [proposals]
    if not args.onnx:
        output = box.predict(inputs)
    else:
        output = box.run(
            None, {k: v for k, v in zip((
                "features_0", "features_1", "features_2", "features_3", "proposals"),
                inputs)})
    class_logits, box_regression = output
    print("4-----------")

    boxes = post_processing(class_logits, box_regression, proposals)
    print("5-----------")

    # track_memory = None
    # inputs = features[:4] + [track_memory]
    # if not args.onnx:
    #     output = tracker.predict(inputs)
    # else:
    #     output = tracker.run(
    #         None, {k: v for k, v in zip((
    #             "features_0", "features_1", "features_2", "features_3",
    #             "boxes", "sr", "template_features"),
    #             inputs)})
    # y, tracks, loss_track = output

    # solver is only needed during inference
    # if tracks is not None:
    #     tracks = self._refine_tracks(features, tracks)
    #     detections = [cat_boxlist(detections + tracks)]

    boxes = solver(boxes)

    # # get the current state for tracking
    # x = get_track_memory(features, detections)

    # cache['x'] = x

    return boxes


def recognize_from_video(rpn, box, tracker):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != None:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    anchors = anchor_generator(
        ((200, 320), (100, 160), (50, 80), (25, 40), (13, 20))
    )

    i = 0
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        output = predict(rpn, box, tracker, img, anchors)

        i += 1
        continue

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
    # dic_model = {
    #     'xxx': (WEIGHT_PATH, MODEL_PATH),
    #     'XXX': (WEIGHT_XXX_PATH, MODEL_XXX_PATH),
    # }
    # weight_path, model_path = dic_model[args.model_type]
    #
    # # model files check and download
    # logger.info('Checking XXX model...')
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    # logger.info('Checking XXX model...')
    # check_and_download_models(WEIGHT_XXX_PATH, MODEL_XXX_PATH, REMOTE_PATH)
    #
    # if args.video or args.detection:
    #     logger.info('Check object detection model...')
    #     check_and_download_models(
    #         WEIGHT_XXX_PATH, MODEL_XXX_PATH, REMOTE_PATH
    #     )

    # load model
    env_id = args.env_id

    # initialize
    if not args.onnx:
        rpn = ailia.Net("rpn.onnx.prototxt", "rpn.onnx", env_id=env_id)
        box = ailia.Net("box.onnx.prototxt", "box.onnx", env_id=env_id)
        tracker = ailia.Net("tracker.onnx.prototxt", "tracker.onnx", env_id=env_id)
    else:
        import onnxruntime
        rpn = onnxruntime.InferenceSession("rpn.onnx")
        box = onnxruntime.InferenceSession("box.onnx")
        tracker = onnxruntime.InferenceSession("tracker.onnx")

    recognize_from_video(rpn, box, tracker)


if __name__ == '__main__':
    main()
