import sys
import time

import cv2
import argparse

import ailia
import mobilenetv3_labels

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402C


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'clock.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MODEL_LISTS = ['small', 'large']

MAX_CLASS_COUNT = 3
SLEEP_TIME = 3


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='ImageNet classification Model'
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
    '-a', '--arch', metavar='ARCH',
    default='small', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS) + ' (default: small)'
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = f'mobilenetv3_{args.arch}.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/mobilenetv3/'


# ======================
# Utils
# ======================
def print_results(preds_ailia):
    preds_ailia = preds_ailia[0]
    top_scores = preds_ailia.argsort()[-1 * MAX_CLASS_COUNT:][::-1]

    print('==============================================================')
    print(f'class_count={MAX_CLASS_COUNT}')
    for idx in range(MAX_CLASS_COUNT):
        print(f'+ idx={idx}')
        print(f'  category={top_scores[idx]}['
              f'{mobilenetv3_labels.imagenet_category[top_scores[idx]]} ]')
        print(f'  prob={preds_ailia[top_scores[idx]]}')


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='ImageNet',
        gen_input_ailia=True
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(input_data)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    print_results(preds_ailia)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

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

        input_image, input_data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='ImageNet'
        )

        # Inference
        preds_ailia = net.predict(input_data)

        print_results(preds_ailia)
        cv2.imshow('frame', input_image)
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
