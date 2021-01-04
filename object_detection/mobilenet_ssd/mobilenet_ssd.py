import sys
import time

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from detector_utils import plot_results  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'annotated.png'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
MODEL_LISTS = ['mb1-ssd', 'mb2-ssd-lite']


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MultiBox Detector', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='mb2-ssd-lite', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS) + ' (default: mb2-ssd-lite)'
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH = args.arch + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mobilenet_ssd/'

VOC_CATEGORY = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    categories = 80
    threshold = 0.4
    iou = 0.45
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        categories,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_SSD,
        env_id=args.env_id,
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        org_img = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='None',
        )
        if org_img.shape[2] == 3:
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2BGRA)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                detector.compute(org_img, threshold, iou)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            detector.compute(org_img, threshold, iou)

        # postprocessing
        res_img = plot_results(detector, org_img, VOC_CATEGORY)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    categories = 80
    threshold = 0.4
    iou = 0.45
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        categories,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_SSD,
        env_id=args.env_id,
    )

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, resized_img = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2BGRA)
        detector.compute(img, threshold, iou)
        res_img = plot_results(detector, resized_img, VOC_CATEGORY, False)
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

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
