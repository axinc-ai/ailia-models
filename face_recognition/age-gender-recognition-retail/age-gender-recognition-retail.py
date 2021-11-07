import sys
import time

import cv2
import numpy as np

import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

sys.path.append('../../face_detection/blazeface')
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'age-gender-recognition-retail-0013.onnx'
MODEL_PATH = 'age-gender-recognition-retail-0013.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/'

FACE_WEIGHT_PATH = 'blazefaceback.onnx'
FACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_IMAGE_HEIGHT = 256
FACE_IMAGE_WIDTH = 256
FACE_MIN_SCORE_THRESH = 0.5

IMAGE_PATH = 'demo.jpg'
IMAGE_SIZE = 62

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'age-gender-recognition', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-d', '--detection',
    action='store_true',
    help='Use face detection.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def recognize_from_image(net, detector):
    # prepare input data
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        if args.detection:
            frame = cv2.imread(image_path)
            recognize_from_frame(net, detector, frame)
            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, frame)
            continue

        img = load_image(
            image_path, (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='None')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = net.predict([img])
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = net.predict([img])

        prob, age_conv3 = output
        prob = prob[0][0][0]
        age_conv3 = age_conv3[0][0][0][0]

        i = np.argmax(prob)
        logger.info(" gender is: %s (%.2f)" % ('Female' if i == 0 else 'Male', prob[i] * 100))
        logger.info(" age is: %d" % round(age_conv3 * 100))

    logger.info('Script finished successfully.')


def recognize_from_frame(net, detector, frame):
    # detect face
    detections = compute_blazeface(
        detector,
        frame,
        anchor_path='../../face_detection/blazeface/anchorsback.npy',
        back=True,
        min_score_thresh=FACE_MIN_SCORE_THRESH
    )

    # adjust face rectangle
    new_detections = []
    for detection in detections:
        margin = 1.5
        r = ailia.DetectorObject(
            category=detection.category,
            prob=detection.prob,
            x=detection.x-detection.w*(margin-1.0)/2,
            y=detection.y-detection.h*(margin-1.0)/2-detection.h*margin/8,
            w=detection.w*margin,
            h=detection.h*margin,
        )
        new_detections.append(r)
    detections = new_detections

    # estimate age and gender
    for obj in detections:
        # get detected face
        margin = 1.0
        crop_img, top_left, bottom_right = crop_blazeface(
            obj, margin, frame
        )
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue

        img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # inference
        output = net.predict([img])
        prob, age_conv3 = output
        prob = prob[0][0][0]
        age_conv3 = age_conv3[0][0][0][0]

        i = np.argmax(prob)
        gender = 'Female' if i == 0 else 'Male'
        age = round(age_conv3 * 100)

        # display label
        LABEL_WIDTH = bottom_right[1] - top_left[1]
        LABEL_HEIGHT = 20
        if gender=="Male":
            color = (255, 128, 128)
        else:
            color = (128, 128, 255)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)
        cv2.rectangle(
            frame,
            top_left,
            (top_left[0] + LABEL_WIDTH, top_left[1] + LABEL_HEIGHT),
            color,
            thickness=-1,
        )

        text_position = (top_left[0], top_left[1] + LABEL_HEIGHT // 2)
        color = (0, 0, 0)
        fontScale = 0.5
        cv2.putText(
            frame,
            "{} {}".format(gender, age),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1,
        )


def recognize_from_video(net, detector):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        recognize_from_frame(net, detector, frame)

        # show result
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
    logger.info('=== age-gender-recognition model ===')
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )
    if args.video:
        logger.info('=== face detection model ===')
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    # load model
    env_id = args.env_id

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )
    detector = None
    if args.video or args.detection:
        detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id)

    # image mode
    if args.video is not None:
        # video mode
        recognize_from_video(net, detector)
    else:
        # image mode
        recognize_from_image(net, detector)


if __name__ == '__main__':
    main()
