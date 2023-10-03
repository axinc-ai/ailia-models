import sys
import time

import ailia
import cv2

import hand_detection_pytorch_utils

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'hand_detection_pytorch.onnx'
MODEL_PATH = 'hand_detection_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hand_detection_pytorch/'

IMAGE_PATH = 'CARDS_OFFICE.jpg'
SAVE_IMAGE_PATH = 'CARDS_OFFICE_output.jpg'

THRESHOLD = 0.2
IOU = 0.2


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'hand-detection.PyTorch hand detection model',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        to_show = imread(image_path, cv2.IMREAD_COLOR)
        logger.info(f'input image shape: {to_show.shape}')
        img, scale = hand_detection_pytorch_utils.pre_process(to_show)
        detector.set_input_shape((1, 3, img.shape[2], img.shape[3]))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                out = detector.predict({'input.1': img})
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            out = detector.predict({'input.1': img})

        dets = hand_detection_pytorch_utils.post_process(
            out, img, scale, THRESHOLD, IOU
        )
        for i in range(dets.shape[0]):
            cv2.rectangle(
                to_show,
                (int(dets[i][0]), int(dets[i][1])),
                (int(dets[i][2]), int(dets[i][3])),
                [0, 0, 255],
                3
            )
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, to_show)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, to_show = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img, scale = hand_detection_pytorch_utils.pre_process(to_show)
        detector.set_input_shape((1, 3, img.shape[2], img.shape[3]))
        out = detector.predict({'input.1': img})
        dets = hand_detection_pytorch_utils.post_process(
            out, img, scale, THRESHOLD, IOU
        )
        for i in range(dets.shape[0]):
            cv2.rectangle(
                to_show,
                (int(dets[i][0]), int(dets[i][1])),
                (int(dets[i][2]), int(dets[i][3])),
                [0, 0, 255],
                3
            )
        cv2.imshow('frame', to_show)
        frame_shown = True
        # save results
        if writer is not None:
            writer.write(to_show)
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
