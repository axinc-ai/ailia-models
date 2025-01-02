import sys
import time

import ailia
import cv2

from mlp_mixer_labels import imagenet_category
from mlp_mixer_utils import *

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/mlp_mixer/"

IMAGE_PATH = 'input.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MAX_CLASS_COUNT = 3
SLEEP_TIME = 0  # for video mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'MLP-Mixer architecture for computer vision', IMAGE_PATH, None
)
parser.add_argument(
    '-m', '--model_name',
    default='Mixer-L_16',
    help='[Mixer-L_16, Mixer-B_16]'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + ".opt.onnx"
MODEL_PATH = MODEL_NAME + ".opt.onnx.prototxt"


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_img = imread(image_path)
        input_data = preprocess_input(input_img, IMAGE_HEIGHT, IMAGE_WIDTH)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds = net.predict(input_data)
            preds = softmax(preds)

        # show results
        print_results(preds, imagenet_category)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # prepare input data
        _, input_data = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = preprocess_input(input_data, IMAGE_HEIGHT, IMAGE_WIDTH)

        # inference
        preds = net.predict(input_data)
        preds = softmax(preds)

        # get result
        plot_results(frame, preds, imagenet_category)

        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
