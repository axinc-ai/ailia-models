import sys
import time
import argparse

import cv2

import ailia
import inceptionv3_labels

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'inceptionv3.onnx'
MODEL_PATH = 'inceptionv3.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/inceptionv3/"

IMAGE_PATH = 'clock.jpg'
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
MAX_CLASS_COUNT = 3
SLEEP_TIME = 3  # for video mode


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Inception architecture for computer vision'
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
    input_img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )
    input_data = cv2.cvtColor(input_img, cv2.COLOR_BGR2BGRA)

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

    # compute execution time
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
    
    # postprocessing
    for idx in range(count):
        # print result
        print(f'+ idx={idx}')
        info = classifier.get_class(idx)
        print(f'  category={info.category}' +
              f'[ {inceptionv3_labels.imagenet_category[info.category]} ]')
        print(f'  prob={info.prob}')
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

        # prepare input data
        input_image, input_data = adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)

        # inference
        classifier.compute(input_data, MAX_CLASS_COUNT)

        # get result
        count = classifier.get_class_count()

        print('==============================================================')
        for idx in range(count):
            # print result
            print(f'+ idx={idx}')
            info = classifier.get_class(idx)
            print(f'  category={info.category}' +
                  f'[ {inceptionv3_labels.imagenet_category[info.category]} ]')
            print(f'  prob={info.prob}')

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
