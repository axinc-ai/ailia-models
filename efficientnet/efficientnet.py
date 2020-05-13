import time
import sys
import argparse
import cv2
import numpy as np

import ailia
import efficientnet_labels

# import original modules
sys.path.append("../util")
from utils import check_file_existance
from model_utils import check_and_download_models
from image_utils import load_image
from webcamera_utils import adjust_frame_size

# ================
# IMAGE PARAMETERS
# ================
IMAGE_PATH = "clock.jpg"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# ======================
# Argument Parser Config
# ======================
parser = argparse.ArgumentParser(
    description = "EfficientNet is "
)
parser.add_argument(
    '-m', '--model', metavar = 'MODEL',
    default = "b7",
    help = "The input model path." +\
           "you can set b0 or b7 to select efficientnet-b0 or efficientnet-b7"
)
parser.add_argument(
    '-i', '--input', metavar = 'IMAGE',
    default = IMAGE_PATH,
    help = "The input image path."
)
parser.add_argument(
    '-v', '--video', metavar = 'VIDEO',
    default = None,
    help = "The input video path." +\
           "If the VIDEO argument is set to 0, the webcam input will be used."
)
args = parser.parse_args()

# ==========================
# MODEL AND OTHER PARAMETERS
# ==========================
MODEL_PATH = "efficientnet-" + args.model + ".onnx.prototxt"
WEIGHT_PATH = "efficientnet-" + args.model + ".onnx"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/efficientnet/"

MAX_CLASS_COUNT = 3
SLEEP_TIME = 3 # for web cam mode

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
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32
    )

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        classifier.compute(input_data, MAX_CLASS_COUNT)
        count = classifier.get_class_count()
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # show results
    print(f'class_count: {count}')
    for idx in range(count):
        print(f'+ idx={idx}')
        info = classifier.get_class(idx)
        print(f'  category={info.category} [ ' +\
              f'{efficientnet_labels.imagenet_category[info.category]} ]')
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
        range=ailia.NETWORK_IMAGE_RANGE_S_FP32
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

        _, resized_frame = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        input_data = cv2.cvtColor(
            resized_frame.astype(np.float32),
            cv2.COLOR_RGB2BGRA
        ).astype(np.uint8)

        classifier.compute(input_data, MAX_CLASS_COUNT)
        count = classifier.get_class_count()

        # show results
        print('==============================================================')
        print(f'class_count: {count}')
        for idx in range(count):
            print(f'+ idx={idx}')
            info = classifier.get_class(idx)
            print(f'  category={info.category} [ ' +\
                  f'{efficientnet_labels.imagenet_category[info.category]} ]')
            print(f'  prob={info.prob}')

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
