import sys
from functools import partial

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
from track_utils import track_utils, track_pool, track_head, track_solver

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

    # boxlists = zip(*sampled_boxes)
    # boxlists = [
    #     [np.concatenate(x, axis=0) for x in zip(*boxlist)] for boxlist in boxlists
    # ]
    boxlists = zip(*sampled_boxes)
    boxlist = [boxes_cat(boxlist) for boxlist in boxlists]

    boxlist = select_over_all_levels(boxlist)
    print("boxlist--", boxlist[0].bbox)
    print("boxlist--", boxlist[0].bbox.shape)

    return boxlist[0]


def post_processing(class_logits, box_regression, bbox, ids=None):
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
    boxes.bbox[:, 0] = boxes.bbox[:, 0].clip(0, max=1280 - 1)
    boxes.bbox[:, 1] = boxes.bbox[:, 1].clip(0, max=800 - 1)
    boxes.bbox[:, 2] = boxes.bbox[:, 2].clip(0, max=1280 - 1)
    boxes.bbox[:, 3] = boxes.bbox[:, 3].clip(0, max=800 - 1)

    boxes = filter_results(boxes, num_classes)

    return boxes


def extract_cache(feat_ext, features, detection):
    """
    Get the cache (state) that is necessary for tracking
    output: (features for tracking targets,
             search region,
             detection bounding boxes)
    """

    # get cache features for search region
    # FPN features
    inputs = features[:4] + [detection.bbox]
    if not args.onnx:
        output = feat_ext.predict(inputs)
    else:
        output = feat_ext.run(
            None, {k: v for k, v in zip((
                "feature_0", "feature_1", "feature_2", "feature_3", "proposals"),
                inputs)})
    x = output[0]
    print("x---", x)
    print("x---", x.shape)

    sr = track_utils.update_boxes_in_pad_images(detection)
    sr = track_utils.extend_bbox(sr)

    cache = (x, sr, detection)

    return cache


def predict(rpn, box, tracker, feat_ext, img, anchors, cache={}):
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

    ### roi_heads.box

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

    ### roi_heads.track

    track_head.feature_extractor = feat_ext

    track_memory = cache.get('x', None)

    track_boxes = None
    if track_memory is None:
        track_pool.reset()
        y, tracks = {}, track_boxes
    else:
        template_features, sr, template_boxes = track_memory
        inputs = features[:4] + [template_features, sr, template_boxes]
        if not args.onnx:
            output = tracker.predict(inputs)
        else:
            output = tracker.run(
                None, {k: v for k, v in zip((
                    "features_0", "features_1", "features_2", "features_3",
                    "boxes", "sr", "template_features"),
                    inputs)})
        cls_logits, center_logits, reg_logits = output

        cls_logits = F.interpolate(cls_logits, scale_factor=16, mode='bicubic')
        center_logits = F.interpolate(center_logits, scale_factor=16, mode='bicubic')
        reg_logits = F.interpolate(reg_logits, scale_factor=16, mode='bicubic')

        locations = get_locations(sr_features, template_features, sr, shift_xy=(shift_x, shift_y), up_scale=16)

        assert len(boxes) == 1
        bb, bb_conf = decode_response(cls_logits, center_logits, reg_logits, locations, boxes[0],
                                      use_centerness=self.use_centerness, sigma=self.sigma)
        track_result = wrap_results_to_boxlist(bb, bb_conf, boxes, amodal=self.amodal)

        y, tracks = {}, track_result

    # solver is only needed during inference
    if tracks is not None:
        tracks = self._refine_tracks(features, tracks)
        boxes = [cat_boxlist(boxes + tracks)]

    boxes = track_solver.solve(boxes)

    # get the current state for tracking
    x = track_head.get_track_memory(
        features, boxes,
        extract_cache=partial(extract_cache, feat_ext))

    cache['x'] = x

    return boxes


def recognize_from_video(rpn, box, tracker, feat_ext):
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

        if i > 2:
            break
        frame = np.load("%03d.npy" % (i + 1))

        # inference
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        output = predict(rpn, box, tracker, feat_ext, img, anchors)

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
        feat_ext = onnxruntime.InferenceSession("feat_ext.onnx")

    recognize_from_video(rpn, box, tracker, feat_ext)


if __name__ == '__main__':
    main()
