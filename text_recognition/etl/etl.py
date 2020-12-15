import sys
import time
import codecs

import cv2

import ailia
# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'etl_BINARY_squeezenet128_20.prototxt'
WEIGHT_PATH = 'etl_BINARY_squeezenet128_20.caffemodel'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/etl/'

IMAGE_PATH = 'font.png'
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
ETL_PATH = 'etl_BINARY_squeezenet128_20.txt'
MAX_CLASS_COUNT = 3
SLEEP_TIME = 0  # for webcam mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Japanese character classification model.', IMAGE_PATH, None
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def preprocess_image(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    img = cv2.bitwise_not(img)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    img = preprocess_image(img)

    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
    )

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            classifier.compute(img, MAX_CLASS_COUNT)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        classifier.compute(img, MAX_CLASS_COUNT)

    # get result
    count = classifier.get_class_count()
    print(f'class_count: {count}')

    for idx in range(count):
        print(f"+ idx={idx}")
        info = classifier.get_class(idx)
        print(f"  category={info.category} [ {etl_word[info.category]} ]")
        print(f"  prob={info.prob}")


def recognize_from_video():
    etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()

    # net initialize
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=args.env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
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

        in_frame, frame = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        frame = preprocess_image(frame)

        # inference
        # compute execution time
        classifier.compute(frame, MAX_CLASS_COUNT)

        # get result
        count = classifier.get_class_count()

        print('==============================================================')
        print(f'class_count: {count}')
        for idx in range(count):
            print(f"+ idx={idx}")
            info = classifier.get_class(idx)
            print(f"  category={info.category} [ {etl_word[info.category]} ]")
            print(f"  prob={info.prob}")
        cv2.imshow('frame', in_frame)
        # save results
        if writer is not None:
            writer.write(in_frame)
        time.sleep(SLEEP_TIME)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    print('Script finished successfully.')


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
