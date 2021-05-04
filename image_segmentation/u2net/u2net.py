import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402
from u2net_utils import load_image, transform, save_result, norm  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 320
MODEL_LISTS = ['small', 'large']
OPSET_LISTS = ['10', '11']


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('U square net', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='large', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
parser.add_argument(
    '-o', '--opset', metavar='OPSET',
    default='11', choices=OPSET_LISTS,
    help='opset lists: ' + ' | '.join(OPSET_LISTS)
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
if args.opset == "10":
    WEIGHT_PATH = 'u2net.onnx' if args.arch == 'large' else 'u2netp.onnx'
else:
    WEIGHT_PATH = 'u2net_opset11.onnx' \
        if args.arch == 'large' else 'u2netp_opset11.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'


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

        # prepare input data
        input_data, h, w = load_image(
            image_path,
            scaled_size=IMAGE_SIZE,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict([input_data])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            # dim = [(1, 1, 320, 320), (1, 1, 320, 320),..., ]  len=7
            preds_ailia = net.predict([input_data])

        # postprocessing
        # we only use `d1` (the first output, check the original repository)
        pred = preds_ailia[0][0, 0, :, :]

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        save_result(pred, savepath, [h, w])

        # composite
        if args.composite:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            image[:, :, 3] = cv2.resize(pred, (w, h)) * 255
            cv2.imwrite(savepath, image)

    logger.info('Script finished successfully.')


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
        #writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, rgb=False) # alpha
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w) # composite
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_data = transform(frame, IMAGE_SIZE)

        # inference
        preds_ailia = net.predict([input_data])

        # postprocessing
        pred = cv2.resize(norm(preds_ailia[0][0, 0, :, :]), (f_w, f_h))

        # force composite
        frame[:, :, 0] = frame[:, :, 0] * pred
        frame[:, :, 1] = frame[:, :, 1] * pred
        frame[:, :, 2] = frame[:, :, 2] * pred
        pred = frame / 255.0

        cv2.imshow('frame', pred)

        # save results
        if writer is not None:
            writer.write((pred * 255).astype(np.uint8))

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
