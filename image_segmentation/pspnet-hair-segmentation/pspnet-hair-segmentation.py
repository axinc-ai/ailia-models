import sys
import time

import numpy as np
import cv2

import ailia
# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'pspnet-hair-segmentation.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH =\
    'https://storage.googleapis.com/ailia-models/pspnet-hair-segmentation/'

IMAGE_PATH = 'test.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Real-time hair segmentation model', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(src_img, preds_ailia):
    pred = sigmoid(preds_ailia)[0][0]
    mask = pred >= 0.5

    mask_n = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    mask_n[:, :, 0] = 255
    mask_n[:, :, 0] *= mask

    image_n = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

    # discard padded area
    h, w, _ = image_n.shape
    delta_h = h - IMAGE_HEIGHT
    delta_w = w - IMAGE_WIDTH

    top = delta_h // 2
    bottom = IMAGE_HEIGHT - (delta_h - top)
    left = delta_w // 2
    right = IMAGE_WIDTH - (delta_w - left)

    mask_n = mask_n[top:bottom, left:right, :]
    image_n = image_n * 0.5 + mask_n * 0.5
    return image_n


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
        src_img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds_ailia = net.predict(input_data)

        # postprocessing
        res_img = postprocess(src_img, preds_ailia)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        src_img, input_data = webcamera_utils.preprocess_frame(
            frame,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            normalize_type='ImageNet'
        )

        src_img = cv2.resize(src_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        preds_ailia = net.predict(input_data)

        res_img = postprocess(src_img, preds_ailia)
        cv2.imshow('frame', res_img / 255.0)

        # # save results
        # if writer is not None:
        #     writer.write(res_img)

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
