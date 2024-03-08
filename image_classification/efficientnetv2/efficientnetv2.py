import sys
import time
import cv2

import ailia
import efficientnetv2_labels

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/efficientnetv2/'

IMAGE_PATH = 'input.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SLEEP_TIME = 0

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('Image classification model: efficientnetv2', IMAGE_PATH, None)
parser.add_argument(
    '--model_name',
    default='efficientnetv2-b0',
    help='[efficientnetv2-b0, efficientnetv2-b1, efficientnetv2-b2, efficientnetv2-b3]'
)
args = update_parser(parser)

MODEL_NAME = args.model_name
WEIGHT_PATH = MODEL_NAME + '.opt.onnx'
MODEL_PATH = MODEL_NAME + '.opt.onnx.prototxt'
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
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='ImageNet',
            gen_input_ailia=True,
        )
        input_data = input_data.transpose(0,2,3,1)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # postprocessing
        print_results(preds_ailia, efficientnetv2_labels.imagenet_category)
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

        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='ImageNet'
        )
        input_data = input_data.transpose(0, 2, 3, 1)

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        plot_results(frame, preds_ailia, efficientnetv2_labels.imagenet_category)
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
