import sys
import time
import argparse

import cv2

import ailia
import vgg16_labels

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame, get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'vgg16_pytorch.onnx'
MODEL_PATH = 'vgg16_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vgg16/'

IMAGE_PATH = 'pizza.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 5
SLEEP_TIME = 3

# TODO
# model_path = "VGG16.prototxt"
# weight_path = "VGG16.caffemodel"
# img_path = './pizza.jpg'


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Image classification model: VGG16'
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
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def print_results(preds_ailia):
    top_scores = preds_ailia[0].argsort()[-1 * MAX_CLASS_COUNT:][::-1]

    print('==============================================================')
    print(f'class_count={MAX_CLASS_COUNT}')
    for idx in range(MAX_CLASS_COUNT):
        print(f'+ idx={idx}')
        print(f'  category={top_scores[idx]}['
              f'{vgg16_labels.imagenet_category[top_scores[idx]]} ]')
        print(f'  prob={preds_ailia[0][top_scores[idx]]}')


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

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(input_data)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(input_data)

    # postprocessing
    print_results(preds_ailia)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, input_data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='ImageNet'
        )

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
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
