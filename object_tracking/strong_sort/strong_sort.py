import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2
from matplotlib import cm

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402

# from bytetrack_utils import multiclass_nms
# from tracker.byte_tracker import BYTETracker

_this = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(os.path.dirname(_this))

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_FRID_PATH = 'duke_bot_S50.onnx'
MODEL_FRID_PATH = 'duke_bot_S50.onnx.prototxt'
REMOTE_FRID_PATH = \
    'https://storage.googleapis.com/ailia-models/strong_sort/'

WEIGHT_MOT17_X_PATH = 'bytetrack_x_mot17.onnx'
MODEL_MOT17_X_PATH = 'bytetrack_x_mot17.onnx.prototxt'
WEIGHT_MOT17_S_PATH = 'bytetrack_s_mot17.onnx'
MODEL_MOT17_S_PATH = 'bytetrack_s_mot17.onnx.prototxt'
WEIGHT_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx'
MODEL_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx.prototxt'
WEIGHT_MOT20_X_PATH = 'bytetrack_x_mot20.onnx'
MODEL_MOT20_X_PATH = 'bytetrack_x_mot20.onnx.prototxt'
REMOTE_BYTRK_PATH = \
    'https://storage.googleapis.com/ailia-models/bytetrack/'

WEIGHT_YOLOX_S_PATH = 'yolox_s.opt.onnx'
MODEL_YOLOX_S_PATH = 'yolox_s.opt.onnx.prototxt'
WEIGHT_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx'
MODEL_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = \
    'https://storage.googleapis.com/ailia-models/yolox/'

VIDEO_PATH = 'demo.mp4'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'StrongSORT', VIDEO_PATH, None
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
    '-m', '--model_type', default='mot17_x',
    choices=('mot17_x', 'mot20_x', 'mot17_s', 'mot17_tiny', 'yolox_s', 'yolox_tiny'),
    help='model type'
)
parser.add_argument(
    '--gui',
    action='store_true',
    help='Display preview in GUI.'
)
# tracking args
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def setup_detector(net):
    sys.path.append(os.path.join(top_path, 'object_tracking/bytetrack'))
    from bytetrack_mod import mod, set_args  # noqa

    set_args(args)

    def _detector(img):
        dets = mod.predict(net, img)
        return dets

    detector = _detector

    return detector


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

def predict(mod, img):
    detector = mod["detector"]
    net = mod["net"]

    dets = detector(img)

    crop_img = [img[d[1]:d[3], d[0]:d[2], :] for d in dets.astype(int)]
    for i, img in enumerate(crop_img):
        cv2.imwrite("kekka%02d.png" % i, img)

    1 / 0

    # feedforward
    output = net.predict([img])
    output = output[0]

    # For yolox, retrieve only the person class
    output = output[..., :6]

    score_thre = args.score_thre
    nms_thre = args.nms_thre
    dets = postprocess(output, ratio, img_size, nms_thre=nms_thre, score_thre=score_thre)

    return dets


def benchmarking(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    _, frame = capture.read()

    logger.info('BENCHMARK mode')

    total_time_estimation = 0
    for i in range(args.benchmark_count):
        start = int(round(time.time() * 1000))
        predict(net, frame)
        end = int(round(time.time() * 1000))
        estimation_time = (end - start)

        # Loggin
        logger.info(f'\tailia processing estimation time {estimation_time} ms')
        if i != 0:
            total_time_estimation = total_time_estimation + estimation_time

    logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')


def recognize_from_video(mod):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # # create video writer if savepath is specified as video format
    # f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # if args.savepath is not None:
    #     writer = get_writer(args.savepath, f_h, f_w)
    # else:
    #     writer = None

    tracker = None

    frame_shown = False
    while True:
        # ret, frame = capture.read()
        # if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
        #     break
        # if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
        #     break
        frame = cv2.imread("input.jpg")

        # inference
        output = predict(mod, frame)

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
        if args.gui or args.video:
            cv2.imshow('frame', res_img)
            frame_shown = True
        else:
            print("Online ids", online_ids)

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
        'mot17_x': (WEIGHT_MOT17_X_PATH, MODEL_MOT17_X_PATH),
        'mot17_s': (WEIGHT_MOT17_S_PATH, MODEL_MOT17_S_PATH),
        'mot17_tiny': (WEIGHT_MOT17_TINY_PATH, MODEL_MOT17_TINY_PATH),
        'mot20_x': (WEIGHT_MOT20_X_PATH, MODEL_MOT20_X_PATH),
        'yolox_s': (WEIGHT_YOLOX_S_PATH, MODEL_YOLOX_S_PATH),
        'yolox_tiny': (WEIGHT_YOLOX_TINY_PATH, MODEL_YOLOX_TINY_PATH),
    }
    model_type = args.model_type
    WEIGHT_PATH, MODEL_PATH = dic_model[model_type]

    # model files check and download
    check_and_download_models(WEIGHT_FRID_PATH, MODEL_FRID_PATH, REMOTE_FRID_PATH)
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH,
        REMOTE_BYTRK_PATH if model_type.startswith('mot') else REMOTE_YOLOX_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    mem_mode = ailia.get_memory_mode(reduce_constant=True, reuse_interstage=True)
    det_net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=mem_mode)
    detector = setup_detector(det_net)

    mod = {
        "detector": detector,
        "net": net,
    }

    if args.benchmark:
        benchmarking(mod)
    else:
        recognize_from_video(mod)


if __name__ == '__main__':
    main()
