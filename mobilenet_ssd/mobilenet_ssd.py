import sys
import time
import argparse

import numpy as np
import cv2

import ailia
from mobilenet_ssd_utils import plot_result

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size  # noqa: E402C


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'annotated.jpg'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
MODEL_LISTS = ['mb1-ssd', 'mb2-ssd-lite']


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='MultiBox Detector'
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
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='mb2-ssd-lite', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS) + ' (default: mb2-ssd-lite)'
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
WEIGHT_PATH = args.arch + '.onnx'
MODEL_PATH = args.arch + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mobilenet_ssd/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )
    input_data = org_img / 255.0
    input_data = input_data.transpose((2, 0, 1))
    input_data = input_data[np.newaxis, :, :, :]

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))

        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        scores, boxes = net.get_results()
        
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    res_img = plot_result(org_img, scores, boxes)
    cv2.imwrite(args.savepath, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
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

        _, resized_img = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
        input_data = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) / 255.0
        input_data = input_data.transpose((2, 0, 1))
        input_data = input_data[np.newaxis, :, :, :]

        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        scores, boxes = net.get_results()
         
        # postprocessing
        res_img = plot_result(resized_img, scores, boxes)
        cv2.imshow('frame', res_img)

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
