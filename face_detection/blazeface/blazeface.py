import sys
import time

import cv2

import ailia
import blazeface_utils as but

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH1 = 'blazeface.onnx'
MODEL_PATH1 = 'blazeface.onnx.prototxt'
WEIGHT_PATH2 = 'blazefaceback.onnx'
MODEL_PATH2 = 'blazefaceback.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'result.png'
IMAGE_HEIGHT1 = 128
IMAGE_WIDTH1 = 128
IMAGE_HEIGHT2 = 256
IMAGE_WIDTH2 = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'BlazeFace is a fast and light-weight face detector.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument('-b', '--back', action='store_true')
args = update_parser(parser)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.back == True:
        net = ailia.Net(MODEL_PATH2, WEIGHT_PATH2, env_id=args.env_id)
        IMAGE_HEIGHT = IMAGE_HEIGHT2
        IMAGE_WIDTH = IMAGE_WIDTH2
    else:
        net = ailia.Net(MODEL_PATH1, WEIGHT_PATH1, env_id=args.env_id)
        IMAGE_HEIGHT = IMAGE_HEIGHT1
        IMAGE_WIDTH = IMAGE_WIDTH1

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        org_img = load_image(image_path, (IMAGE_HEIGHT, IMAGE_WIDTH))

        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='127.5',
            gen_input_ailia=True
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                preds_ailia = net.predict([input_data])
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            preds_ailia = net.predict([input_data]) 

        # post-processing
        if args.back == True:
            detections = but.postprocess(preds_ailia, args.back, anchor_path='anchorsback.npy')
        else:
            detections = but.postprocess(preds_ailia, args.back)

        # generate detections
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        for detection in detections:
            but.plot_detections(org_img, detection, save_image_path=savepath)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.back == True:
        net = ailia.Net(MODEL_PATH2, WEIGHT_PATH2, env_id=args.env_id)
        IMAGE_HEIGHT = IMAGE_HEIGHT2
        IMAGE_WIDTH = IMAGE_WIDTH2
    else:
        net = ailia.Net(MODEL_PATH1, WEIGHT_PATH1, env_id=args.env_id)
        IMAGE_HEIGHT = IMAGE_HEIGHT1
        IMAGE_WIDTH = IMAGE_WIDTH1

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5'
        )

        # inference
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()

        # postprocessing
        detections = but.postprocess(preds_ailia)
        but.show_result(input_image, detections)

        # remove padding
        dh = input_image.shape[0]
        dw = input_image.shape[1]
        sh = frame.shape[0]
        sw = frame.shape[1]
        input_image = input_image[(dh-sh)//2:(dh-sh)//2+sh,(dw-sw)//2:(dw-sw)//2+sw,:]

        cv2.imshow('frame', input_image)

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
    if args.back == True:
        check_and_download_models(WEIGHT_PATH2, MODEL_PATH2, REMOTE_PATH)
        
    else:
        check_and_download_models(WEIGHT_PATH1, MODEL_PATH1, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
