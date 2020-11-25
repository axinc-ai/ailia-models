import sys
import time
import argparse

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'pytorch-unet.onnx'
MODEL_PATH = 'pytorch-unet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pytorch-unet/'

IMAGE_PATH = 'data/imgs/0cdf5b5d0ce1_14.jpg'
SAVE_IMAGE_PATH = 'data/masks/0cdf5b5d0ce1_14.jpg'

IMAGE_WIDTH = 959
IMAGE_HEIGHT = 640

THRESHOLD = 0.5


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='hand-detection.PyTorch hand detection model'
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
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '--onnx',
    default=True,
    action='store_true',
    help='execute onnxruntime version.'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def load_image(input_path):
    return np.array(Image.open(input_path))


def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
    img = np.expand_dims(img, 0)
    img_trans = img.transpose((0, 3, 1, 2)) # NHWC to NCHW
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans.astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def post_process(output):
    probs = sigmoid(output)
    probs = probs.squeeze(0)
    full_mask = cv2.resize(
        probs.transpose(1, 2, 0), 
        (IMAGE_WIDTH, IMAGE_HEIGHT), 
        interpolation = cv2.INTER_CUBIC
    )
    mask = full_mask > THRESHOLD
    return mask.astype(np.uint8)*255


def segment_image(img, net):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        out = net.predict({'input.1': img})
    else:
        first_input_name = net.get_inputs()[0].name
        first_output_name = net.get_outputs()[0].name
        output = net.run([first_output_name], {first_input_name: img})[0]

    out = post_process(output)
    return out


# ======================
# Main functions
# ======================
def recognize_from_image(net):
    # prepare input data
    img = load_image(args.input)
    print(f'input image shape: {img.shape}')

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            out = segment_image(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        out = segment_image(img, net)

    cv2.imwrite(args.savepath, out)
    print('Script finished successfully.')


def recognize_from_video(net):
    capture = get_capture(args.video)
    while(True):
        ret, img = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        out = segment_image(img, net)
        cv2.imshow('frame', out)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    # model initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
