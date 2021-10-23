import sys
import time

import numpy as np
import cv2
from matplotlib import cm

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

from bytetrack_utils import multiclass_nms
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

IMAGE_PATH = 'demo.mp4'

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ByteTrack', IMAGE_PATH, None
)
parser.add_argument(
    "--score_thre", type=float, default=0.1,
    help="Score threshould to filter the result.",
)
parser.add_argument(
    "--nms_thre", type=float, default=0.7,
    help="NMS threshould.",
)
parser.add_argument(
    "--with_p6",
    action="store_true",
    help="Whether your model uses p6 in FPN/PAN.",
)
parser.add_argument(
    '-m', '--model_type', default='xxx', choices=('xxx', 'XXX'),
    help='model type'
)
# tracking args
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

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


def frame_vis_generator(frame, bboxes, ids):
    for i, entity_id in enumerate(ids):
        color = vis_colors[int(entity_id) % num_colors]

        x1, y1, w, h = np.round(bboxes[i]).astype(int)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

    return frame


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


def postprocess(output, ratio, p6=False, nms_thre=0.7, score_thre=0.1):
    img_size = (800, 1440)

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    output[..., :2] = (output[..., :2] + grids) * expanded_strides
    output[..., 2:4] = np.exp(output[..., 2:4]) * expanded_strides

    predictions = output[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thre, score_thr=score_thre)

    return dets[:, :-1]


def predict(net, img):
    shape = (800, 1440)
    img, ratio = preprocess(img, shape)

    # feedforward
    output = net.predict([img])
    output = output[0]

    with_p6 = args.with_p6
    score_thre = args.score_thre
    nms_thre = args.nms_thre
    dets = postprocess(output, ratio, nms_thre=nms_thre, score_thre=score_thre, p6=with_p6)

    return dets


def recognize_from_image(net):
    min_box_area = args.min_box_area

    tracker = BYTETracker(
        track_thresh=args.track_thresh, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh, frame_rate=30,
        mot20=args.mot20)

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
        print("online_tlwhs--", online_tlwhs)
        print("online_ids--", online_ids)
        print("online_ids--", len(online_ids))
        print("online_scores--", online_scores)
        print("online_scores--", len(online_scores))

        res_img = frame_vis_generator(img, online_tlwhs, online_ids)

        # plot result
        cv2.imwrite(args.savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
    min_box_area = args.min_box_area

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

    tracker = BYTETracker(
        track_thresh=args.track_thresh, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh, frame_rate=30,
        mot20=args.mot20)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # inference
        output = predict(net, frame)

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

        res_img = frame_vis_generator(frame, online_tlwhs, online_ids)

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
