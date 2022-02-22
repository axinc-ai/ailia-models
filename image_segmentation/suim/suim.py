import sys
import time
import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402 noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/suim/'

WEIGHT_PATH = "suim.opt.onnx"
MODEL_PATH = "suim.opt.onnx.prototxt"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'
HEIGHT = 256
WIDTH = 320


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('suim model', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        img = Image.open(image_path)
        img = np.array(img) / 255.
        logger.debug(f'input image shape: {img.shape}')
        img = cv2.resize(img, (WIDTH, HEIGHT))

        img = np.expand_dims(img, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                pred = net.predict(img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            pred = net.predict(img)

        # postprocessing
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.


        # save individual output masks
        ROs = np.reshape(pred[0, :, :, 0], (HEIGHT, WIDTH))
        FVs = np.reshape(pred[0, :, :, 1], (HEIGHT, WIDTH))
        HDs = np.reshape(pred[0, :, :, 2], (HEIGHT, WIDTH))
        RIs = np.reshape(pred[0, :, :, 3], (HEIGHT, WIDTH))
        WRs = np.reshape(pred[0, :, :, 4], (HEIGHT, WIDTH))

        # save
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')

        Image.fromarray(np.uint8(ROs * 255.)).save('output/' + 'ROs' + savepath)
        Image.fromarray(np.uint8(FVs * 255.)).save('output/' + 'FVs' + savepath)
        Image.fromarray(np.uint8(HDs * 255.)).save('output/' + 'HDs_' + savepath)
        Image.fromarray(np.uint8(RIs * 255.)).save('output/' + 'RIs_' + savepath)
        Image.fromarray(np.uint8(WRs * 255.)).save('output/' + 'WRs_' + savepath)

        if cv2.waitKey(0) != 32:  # space bar
            exit()


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, rgb=False)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input = cv2.resize(frame, (WIDTH, HEIGHT))/ 255.
        input = np.expand_dims(input, 0)

        # inference
        pred = net.predict(input)

        # postprocessing
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.

        # save individual output masks
        HDs = np.reshape(pred[0, :, :, 2], (HEIGHT, WIDTH))

        cv2.imshow('frame', HDs)

        # save results
        if writer is not None:
            writer.write(HDs)

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
