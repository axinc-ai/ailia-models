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
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

from this_utils import anchor_generator, box_decode
from this_utils import remove_small_boxes, box_nms

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
    num_levels = len(objectness)
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
        # o = np.take(o, topk_idx[0], axis=1)
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
        for proposal, scores in zip(proposals, o):
            proposal = remove_small_boxes(proposal, min_size)
            proposal = box_nms(
                proposal,
                scores,
                nms_thresh,
                max_proposals=post_nms_top_n,
            )
            print("boxlist----", proposal.shape)
            result.append(proposal)
    1 / 0

    #     sampled_boxes.append(result)
    #
    # boxlists = zip(*sampled_boxes)
    # boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
    #
    # if num_levels > 1:
    #     boxlists = select_over_all_levels(boxlists)


def post_processing(pixel_pos_scores, link_pos_scores, image_shape):
    mask = decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    bboxes = mask_to_bboxes(mask, image_shape)

    return bboxes


def predict(rpn, box, tracker, img, anchors):
    # shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    # img = preprocess(img, shape)

    # feedforward
    if not args.onnx:
        output = rpn.predict([img])
    else:
        output = rpn.run(None, {'img': img})

    features = output[:5]
    objectness = output[5:10]
    rpn_box_regression = output[10:]
    # print("features--", features[0].shape)
    # print("objectness--", objectness[0].shape)
    # print("rpn_box_regression--", rpn_box_regression[0].shape)
    rpn_post_processing(anchors, objectness, rpn_box_regression)

    # bboxes = post_processing(pixel_pos_scores, link_pos_scores, img.shape)

    return


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
