import sys
import time

import cv2

import ailia
import inceptionv3_labels

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'inceptionv3.onnx'
MODEL_PATH = 'inceptionv3.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/inceptionv3/"

IMAGE_PATH = 'clock.jpg'
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
MAX_CLASS_COUNT = 3
SLEEP_TIME = 0  # for video mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Inception architecture for computer vision', IMAGE_PATH, None
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )
        input_data = cv2.cvtColor(input_img, cv2.COLOR_BGR2BGRA)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                classifier.compute(input_data, MAX_CLASS_COUNT)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            classifier.compute(input_data, MAX_CLASS_COUNT)

        count = classifier.get_class_count()

        # postprocessing
        for idx in range(count):
            # logger.info result
            logger.info(f'+ idx={idx}')
            info = classifier.get_class(idx)
            logger.info(
                f'  category={info.category}'
                f'[ {inceptionv3_labels.imagenet_category[info.category]} ]'
            )
            logger.info(f'  prob={info.prob}')
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # prepare input data
        input_image, input_data = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inference
        classifier.compute(input_data, MAX_CLASS_COUNT)

        # get result
        count = classifier.get_class_count()

        logger.info('=' * 80)
        for idx in range(count):
            # logger.info result
            logger.info(f'+ idx={idx}')
            info = classifier.get_class(idx)
            logger.info(
                f'  category={info.category}'
                f'[ {inceptionv3_labels.imagenet_category[info.category]} ]'
            )
            logger.info(f'  prob={info.prob}')

        cv2.imshow('frame', input_image)
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(input_image)

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
