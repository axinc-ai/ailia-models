import sys
from functools import partial

import numpy as np
import cv2
from PIL import Image
from matplotlib import cm

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

from this_utils import BBox, sigmoid, anchor_generator, box_decode
from this_utils import remove_small_boxes, boxes_nms, boxes_cat, boxes_filter
from this_utils import select_over_all_levels, filter_results
from track_utils import track_forward, track_utils, track_head, track_solver

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'xxx.onnx'
MODEL_PATH = 'xxx.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/xxx/'

VIDEO_PATH = 'Cars-1900.mp4'

IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1280

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'XXX', VIDEO_PATH, None
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

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.reshape(N, -1, C, H, W)
    layer = layer.transpose(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def get_colors(n, colormap="gist_ncar"):
    # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
    # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
    # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
    # For more options see:
    # https://matplotlib.org/examples/color/colormaps_reference.html
    # and https://matplotlib.org/users/colormaps.html

    colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
    # Randomly shuffle the colors
    np.random.shuffle(colors)
    # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
    # Also multiply by 255 since cm returns values in the range [0, 1]
    colors = colors[:, (2, 1, 0)] * 255

    return colors


num_colors = 50
vis_colors = get_colors(num_colors)


def frame_vis_generator(frame, results: BBox):
    ids = results.ids
    results = boxes_filter(results, ids >= 0)
    bbox = results.bbox
    ids = results.ids.tolist()
    labels = results.labels.tolist()

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

    for i, entity_id in enumerate(ids):
        color = vis_colors[int(entity_id) % num_colors]
        class_name = class_names[int(labels[i]) - 1]
        text_width = len(class_name) * 20
        x1, y1, x2, y2 = (np.round(bbox[i, :])).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

        # Draw black background rectangle for test
        cv2.rectangle(frame, (x1 - 5, y1 - 25), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, '{}'.format(class_name), (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

    return frame


# ======================
# Main functions
# ======================

def preprocess(img):
    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH)
    im_h, im_w, _ = img.shape

    if im_h > im_w:
        ow = (h * im_w) // im_h
        oh = h
    else:
        oh = (w * im_h) // im_w
        ow = w
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    img = normalize_image(img, normalize_type='ImageNet')

    # padding
    new_img = np.zeros((h, w, 3))
    x = (w - ow) // 2
    y = (h - oh) // 2
    new_img[y:y + oh, x:x + ow, :] = img
    img = new_img

    img = img.transpose(2, 0, 1)  # HWC -> CHW
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

            boxes.bbox[:, 0] = boxes.bbox[:, 0].clip(0, max=IMAGE_WIDTH - 1)
            boxes.bbox[:, 1] = boxes.bbox[:, 1].clip(0, max=IMAGE_HEIGHT - 1)
            boxes.bbox[:, 2] = boxes.bbox[:, 2].clip(0, max=IMAGE_WIDTH - 1)
            boxes.bbox[:, 3] = boxes.bbox[:, 3].clip(0, max=IMAGE_HEIGHT - 1)

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

    return boxlist[0]


def post_processing(
        class_logits, box_regression, bbox,
        ids=None, labels=None):
    prob = softmax(class_logits, -1)

    proposals = box_decode(
        box_regression, bbox,
        weights=(10.0, 10.0, 5.0, 5.0)
    )

    num_classes = prob.shape[1]

    # deafult id is -1
    ids = ids if ids is not None else np.zeros(len(bbox)) - 1

    # this only happens for tracks
    if labels is not None and 0 < len(labels):
        # tracks
        track_inds = np.nonzero(ids >= 0)[0]

        # avoid track bbs be suppressed during nms
        if 0 < len(track_inds):
            prob_cp = np.array(prob)
            prob[track_inds, :] = 0.
            prob[track_inds, labels] = prob_cp[track_inds, labels] + 1.

    boxes = BBox(
        bbox=proposals.reshape(-1, 4),
        scores=prob.reshape(-1),
        ids=ids
    )
    boxes.bbox[:, 0] = boxes.bbox[:, 0].clip(0, max=IMAGE_WIDTH - 1)
    boxes.bbox[:, 1] = boxes.bbox[:, 1].clip(0, max=IMAGE_HEIGHT - 1)
    boxes.bbox[:, 2] = boxes.bbox[:, 2].clip(0, max=IMAGE_WIDTH - 1)
    boxes.bbox[:, 3] = boxes.bbox[:, 3].clip(0, max=IMAGE_HEIGHT - 1)

    boxes = filter_results(boxes, num_classes)

    return boxes


def refine_tracks(net, features, tracks):
    """
    Use box head to refine the bounding box location
    The final vis score is an average between appearance and matching score
    """
    if len(tracks[0].bbox) == 0:
        return tracks[0]

    track_scores = tracks[0].scores + 1.

    proposals = tracks[0].bbox.astype(np.float32)
    inputs = features[:4] + [proposals]
    if not args.onnx:
        output = net.predict(inputs)
    else:
        output = net.run(
            None, {k: v for k, v in zip((
                "feature_0", "feature_1", "feature_2", "feature_3", "proposals"),
                inputs)})
    class_logits, box_regression = output
    tracks = post_processing(
        class_logits, box_regression, proposals,
        tracks[0].ids, tracks[0].labels)

    det_scores = tracks.scores
    det_boxes = tracks.bbox

    scores = (det_scores + track_scores) / 2.
    boxes = det_boxes
    r_tracks = BBox(
        bbox=boxes,
        scores=scores,
        ids=tracks.ids,
        labels=tracks.labels,
    )
    return [r_tracks]


def predict(rpn, box, tracker, feat_ext, img, anchors, cache={}):
    img = preprocess(img)

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
    boxes = rpn_post_processing(anchors, objectness, rpn_box_regression)
    print("3-----------")

    ### roi_heads.box

    proposals = boxes.bbox
    inputs = features[:4] + [proposals]
    if not args.onnx:
        output = box.predict(inputs)
    else:
        output = box.run(
            None, {k: v for k, v in zip((
                "feature_0", "feature_1", "feature_2", "feature_3", "proposals"),
                inputs)})
    class_logits, box_regression = output
    print("4-----------")

    boxes = post_processing(class_logits, box_regression, proposals)
    print("5-----------")

    ### roi_heads.track

    track_memory = cache.get('x', None)
    y, tracks = track_forward(tracker, features, track_memory, args.onnx)

    if tracks is not None:
        tracks = refine_tracks(box, features, tracks)
        boxes = boxes_cat([boxes] + tracks)

    boxes = track_solver.solve(boxes)

    # get the current state for tracking
    track_head.feature_extractor = feat_ext
    x = track_head.get_track_memory(
        feat_ext, features, boxes, args.onnx)

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

        _frame = frame

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = predict(rpn, box, tracker, feat_ext, img, anchors)

        i += 1
        # continue

        new_frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        h, w = _frame.shape[:2]
        pad_h = (IMAGE_HEIGHT - h) // 2
        new_frame[pad_h:pad_h + h, :, :] = _frame
        frame = new_frame

        res_img = frame_vis_generator(frame, boxes)

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
