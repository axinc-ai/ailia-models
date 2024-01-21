import sys

import ailia
import cv2

sys.path.append("../../util")
from logging import getLogger  # noqa: E402

from arg_utils import get_base_parser, update_parser
from detector_utils import plot_results, reverse_letterbox
from model_utils import check_and_download_models  # noqa: E402

from layout_parsing_utils import pdf_to_images, preprocess
from yolox import YOLOX_LABEL_MAP

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/unstructured-inference/layout-parsing"
PDF_PATH = "sample.pdf"
SAVE_IMAGE_PATH = ""
WEIGHT_PATH = "layout_parsing_yolox.onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
INPUT_SHAPE = (1024, 768)
SCORE_THR = 0.25
NMS_THR = 0.1

logger = getLogger(__name__)


parser = get_base_parser('Layout parsing', PDF_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-i", "--input",
    type=str,
    help="input PDF file name",
    default=PDF_PATH,
)
parser.add_argument(
    "-d", "--dpi", default=200, type=int, help="dpi"
)
parser.add_argument(
    '-dt', '--detector',
    action='store_true',
    help='Use detector API (require ailia SDK 1.2.9).'
)
parser.add_argument(
    '-th', '--threshold',
    default=SCORE_THR, type=float,
    help='The detection threshold for yolo. (default: '+str(SCORE_THR)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=NMS_THR, type=float,
    help='The detection iou for yolo. (default: '+str(NMS_THR)+')'
)
args = update_parser(parser)


def infer(detector: ailia.Detector):
    image_names = pdf_to_images(args.input[0], dpi=args.dpi, output_folder=args.savepath)

    for image_name in image_names:
        img_orig = cv2.imread(image_name)
        img_processed, ratio = preprocess(img_orig, INPUT_SHAPE)

        def compute():
            if args.detector:
                detector.compute(img_orig, args.threshold, args.iou)
                return None
            else:
                return detector.run(img_processed)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = compute()
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            output = compute()

        if args.detector:
            res_img = plot_results(detector, img_orig, YOLOX_LABEL_MAP)
            detect_object = detector
        else:
            predictions = postprocess(output[0], INPUT_SHAPE)[0]
            detect_object = predictions_to_object(predictions, img_orig, ratio, args.iou, args.threshold)
            detect_object = reverse_letterbox(detect_object, img_orig, (img_orig.shape[0], img_orig.shape[1]))
            res_img = plot_results(detect_object, img_orig, YOLOX_LABEL_MAP)

        cv2.imwrite(image_name.replace(".ppm", "_parsed.jpg"), res_img)


if __name__ == "__main__":
    # weight_path, model_path, params = get_params()
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    env_id = args.env_id
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(YOLOX_LABEL_MAP),
        format=ailia.NETWORK_IMAGE_FORMAT_BGR,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_INT8,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOX,
        env_id=env_id,
    )
    infer(detector)
