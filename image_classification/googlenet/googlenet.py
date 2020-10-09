import time
import sys
import argparse

import numpy as np
import cv2

import ailia
import googlenet_labels

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'googlenet.onnx'
MODEL_PATH = 'googlenet.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/googlenet/"

IMAGE_PATH = 'pizza.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 3
SLEEP_TIME = 3  # for webcam mode


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='GoogLeNet is a CNN architecture that won ImageNet2014'
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
# Main functions
# ======================
def recognize_from_image():
    # prepare input data

    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
        gen_input_ailia=False
    )
    input_data = cv2.cvtColor(
        input_data.astype(np.float32),
        cv2.COLOR_RGB2BGRA
    ).astype(np.uint8)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    classifier = ailia.Classifier(
        MODEL_PATH,
        WEIGHT_PATH,
        env_id=env_id,
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32
    )

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            classifier.compute(input_data, MAX_CLASS_COUNT)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        classifier.compute(input_data, MAX_CLASS_COUNT)

    count = classifier.get_class_count()

    # show results
    print_results(classifier, googlenet_labels.imagenet_category)

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
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32
    )

    capture = get_capture(args.video)

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, resized_frame = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        input_data = cv2.cvtColor(
            resized_frame.astype(np.float32),
            cv2.COLOR_RGB2BGRA
        ).astype(np.uint8)

        classifier.compute(input_data, MAX_CLASS_COUNT)
        count = classifier.get_class_count()

        # show results
        plot_results(frame, classifier, googlenet_labels.imagenet_category)

        cv2.imshow('frame', frame)
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
