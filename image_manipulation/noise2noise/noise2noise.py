import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'noise2noise_gaussian.onnx'
MODEL_PATH = 'noise2noise_gaussian.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/noise2noise/'

IMAGE_PATH = 'monarch-gaussian-noisy.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Noise2Noise'
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
    '-add_noise', action='store_true',
    help='If add this argument, add noise to input image (which will be saved)'
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
def add_noise(img, noise_param=50):
    height, width = img.shape[0], img.shape[1]
    std = np.random.uniform(0, noise_param)
    noise = np.random.normal(0, std, (height, width, 3))
    noise_img = np.array(img) + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    img = noise_img
    cv2.imwrite('noise_image.png', img)  # TODO make argument for savepath ?
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
    )

    if args.add_noise:
        img = add_noise(img)

    img = img / 255.0
    input_data = img.transpose(2, 0, 1)
    input_data.shape = (1, ) + input_data.shape

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
    output_img = preds_ailia[0].transpose(1, 2, 0) * 255
    output_img = np.clip(output_img, 0, 255)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.savepath, output_img)
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

        _, resized_image = adjust_frame_size(frame, IMAGE_HEIGHT, IMAGE_WIDTH)

        # add noise
        resized_image = add_noise(resized_image)

        resized_image = resized_image / 255.0
        input_data = resized_image.transpose(2, 0, 1)
        input_data.shape = (1, ) + input_data.shape

        # inference
        preds_ailia = net.predict(input_data)

        # side by side
        preds_ailia[:,:,:,0:input_data.shape[3]//2] = input_data[:,:,:,0:input_data.shape[3]//2]

        # postprocessing
        output_img = preds_ailia[0].transpose(1, 2, 0)
        cv2.imshow('frame', output_img)

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
