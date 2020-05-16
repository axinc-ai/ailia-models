import time
import sys
import argparse

import cv2

import ailia
import resnet50_labels

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402C


# ======================
# Parameters 1
# ======================
MODEL_NAMES = ['resnet50.opt', 'resnet50', 'resnet50_pytorch']
IMAGE_PATH = 'pizza.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_RANGE = ailia.NETWORK_IMAGE_RANGE_S_INT8

MAX_CLASS_COUNT = 5
SLEEP_TIME = 3


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Resnet50 ImageNet classification model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
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
    '--arch', '-a', metavar='ARCH',
    default='resnet50.opt', choices=MODEL_NAMES,
    help=('model architecture: ' + ' | '.join(MODEL_NAMES) +
          ' (default: resnet50.opt)')
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH = args.arch + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/resnet50/'


# ======================
# Utils
# ======================
def preprocess_image(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    img = preprocess_image(img)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=IMAGE_RANGE
    )

    # compute execution time
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
        print(f"  category={info.category} " +
              f"[ {resnet50_labels.imagenet_category[info.category]} ]")
        print(f"  prob={info.prob}")
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=IMAGE_RANGE
    )

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        in_frame, frame = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        frame = preprocess_image(frame)

        # inference
        classifier.compute(frame, MAX_CLASS_COUNT)

        # get result
        count = classifier.get_class_count()

        print('==============================================================')
        print(f'class_count: {count}')
        for idx in range(count):
            print(f"+ idx={idx}")
            info = classifier.get_class(idx)
            print(f"  category={info.category} " +
                  f"[ {resnet50_labels.imagenet_category[info.category]} ]")
            print(f"  prob={info.prob}")
        cv2.imshow('frame', in_frame)
        time.sleep(SLEEP_TIME)

    capture.release()
    cv2.destroyAllWindows()
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
