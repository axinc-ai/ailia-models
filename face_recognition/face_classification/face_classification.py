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
SLEEP_TIME = 0

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

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.0
sys.path.append('../../face_detection/blazeface')
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402

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
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # detect face
        detections = compute_blazeface(detector, frame, anchor_path='../../face_detection/blazeface/anchors.npy')

        for obj in detections:
            # get detected face
            crop_img, top_left, bottom_right = crop_blazeface(obj, FACE_MARGIN, frame)
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                continue
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
                
            # emotion inference
            emotion_classifier.compute(crop_img, EMOTION_MAX_CLASS_COUNT)
            count = emotion_classifier.get_class_count()
            print('===========================================================')
            print(f'emotion_class_count={count}')

            # print result
            emotion_text = ""
            for idx in range(count):
                print(f'+ idx={idx}')
                info = emotion_classifier.get_class(idx)
                print(
                    f'  category={info.category} ' +
                    f'[ {EMOTION_CATEGORY[info.category]} ]'
                )
                print(f'  prob={info.prob}')
                if idx == 0:
                    emotion_text = f'[ {EMOTION_CATEGORY[info.category]} ] prob={info.prob:.3f}'
            print('')

            # gender inference
            gender_text = ""
            gender_classifier.compute(crop_img, GENDER_MAX_CLASS_COUNT)
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
                if idx == 0:
                    gender_text = f'[ {GENDER_CATEGORY[info.category]} ] prob={info.prob:.3f}'
            print('')

            # display label
            LABEL_WIDTH = 400
            LABEL_HEIGHT = 20
            color = (255,255,255)
            cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)
            cv2.rectangle(frame, top_left, (top_left[0]+LABEL_WIDTH,top_left[1]+LABEL_HEIGHT), color, thickness=-1)

            text_position = (top_left[0], top_left[1]+LABEL_HEIGHT//2)
            color = (0,0,0)
            fontScale = 0.5
            cv2.putText(
                frame,
                emotion_text + " " + gender_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                1
            )

            # show result
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
    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
