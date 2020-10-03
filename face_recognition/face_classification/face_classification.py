import sys
import time
import argparse

import cv2

import ailia
# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402


# ======================
# PARAMETERS
# ======================
EMOTION_WEIGHT_PATH = 'emotion_miniXception.caffemodel'
EMOTION_MODEL_PATH = 'emotion_miniXception.prototxt'
GENDER_WEIGHT_PATH = "gender_miniXception.caffemodel"
GENDER_MODEL_PATH = "gender_miniXception.prototxt"
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/face_classification/'

IMAGE_PATH = 'lenna.png'
EMOTION_MAX_CLASS_COUNT = 3
GENDER_MAX_CLASS_COUNT = 2
SLEEP_TIME = 3

EMOTION_CATEGORY = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]
GENDER_CATEGORY = ["female", "male"]


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Face Classificaiton Model (emotion & gender)'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input
    # load input image and convert to BGRA
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    emotion_classifier = ailia.Classifier(
        EMOTION_MODEL_PATH,
        EMOTION_WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST
    )
    gender_classifier = ailia.Classifier(
        GENDER_MODEL_PATH,
        GENDER_WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST
    )

    # inference emotion
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            emotion_classifier.compute(img, EMOTION_MAX_CLASS_COUNT)
            end = int(round(time.time() * 1000))
            print(f'\t[EMOTION MODEL] ailia processing time {end - start} ms')
    else:
        emotion_classifier.compute(img, EMOTION_MAX_CLASS_COUNT)
    count = emotion_classifier.get_class_count()
    print(f'emotion_class_count={count}')

    # print result
    for idx in range(count):
        print(f'+ idx={idx}')
        info = emotion_classifier.get_class(idx)
        print(
            f'  category={info.category} [ {EMOTION_CATEGORY[info.category]} ]'
        )
        print(f'  prob={info.prob}')
    print('')

    # inference gender
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            gender_classifier.compute(img, GENDER_MAX_CLASS_COUNT)
            end = int(round(time.time() * 1000))
            print(f'\t[EMOTION MODEL] ailia processing time {end - start} ms')
    else:
        gender_classifier.compute(img, GENDER_MAX_CLASS_COUNT)
    count = gender_classifier.get_class_count()
    print(f'gender_class_count={count}')

    # print reuslt
    for idx in range(count):
        print(f'+ idx={idx}')
        info = gender_classifier.get_class(idx)
        print(
            f'  category={info.category} [ {GENDER_CATEGORY[info.category]} ]'
        )
        print(f'  prob={info.prob}')
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    emotion_classifier = ailia.Classifier(
        EMOTION_MODEL_PATH,
        EMOTION_WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST
    )
    gender_classifier = ailia.Classifier(
        GENDER_MODEL_PATH,
        GENDER_WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST
    )

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # emotion inference
        emotion_classifier.compute(frame, EMOTION_MAX_CLASS_COUNT)
        count = emotion_classifier.get_class_count()
        print('===========================================================')
        print(f'emotion_class_count={count}')

        # print result
        for idx in range(count):
            print(f'+ idx={idx}')
            info = emotion_classifier.get_class(idx)
            print(
                f'  category={info.category} ' +
                f'[ {EMOTION_CATEGORY[info.category]} ]'
            )
            print(f'  prob={info.prob}')
        print('')

        # gender inference
        gender_classifier.compute(frame, GENDER_MAX_CLASS_COUNT)
        count = gender_classifier.get_class_count()
        # print reuslt
        for idx in range(count):
            print(f'+ idx={idx}')
            info = gender_classifier.get_class(idx)
            print(
                f'  category={info.category} ' +
                f'[ {GENDER_CATEGORY[info.category]} ]'
            )
            print(f'  prob={info.prob}')
        print('')
        cv2.imshow('frame', frame)
        time.sleep(SLEEP_TIME)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        EMOTION_WEIGHT_PATH, EMOTION_MODEL_PATH, REMOTE_PATH
    )
    check_and_download_models(
        GENDER_WEIGHT_PATH, GENDER_MODEL_PATH, REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
