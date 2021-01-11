import sys
import time

import cv2
import numpy as np

import ailia
import blazepalm_utils as but

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'person_with_hands.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'BlazePalm, on-device real-time palm detection.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'blazepalm'
# if args.normal:
WEIGHT_PATH = f'{MODEL_NAME}.onnx'
MODEL_PATH = f'{MODEL_NAME}.onnx.prototxt'
# else:
#     WEIGHT_PATH = f'{MODEL_NAME}.opt.onnx'
#     MODEL_PATH = f'{MODEL_NAME}.opt.onnx.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def display_result(img, detections, with_keypoints=True):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]

        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img


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
        src_img = cv2.imread(image_path)
        img256, _, scale, pad = but.resize_pad(src_img[:, :, ::-1])
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                preds = net.predict([input_data])
                normalized_detections = but.postprocess(preds)[0]
                detections = but.denormalize_detections(
                    normalized_detections, scale, pad
                )
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            preds = net.predict([input_data])
            normalized_detections = but.postprocess(preds)[0]
            detections = but.denormalize_detections(
                normalized_detections, scale, pad
            )

        # postprocessing
        display_result(src_img, detections)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
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

        img256, _, scale, pad = but.resize_pad(frame[:, :, ::-1])
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)

        # inference
        preds = net.predict([input_data])
        normalized_detections = but.postprocess(preds)[0]
        detections = but.denormalize_detections(
            normalized_detections, scale, pad
        )

        # postprocessing
        display_result(frame, detections)
        cv2.imshow('frame', frame)

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
