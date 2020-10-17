import sys
import time
import argparse

from PIL import Image
import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
sys.path.append('./util')
import webcamera_utils  # noqa: E402C
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402

# ======================
# Parameters
# ======================

WEIGHT_PATH = './sem_seg.onnx'
MODEL_PATH = './sem_seg.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/semantic-segmentation-with-mobilenet-v3/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY = (
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes',
    'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face',
    'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Semantic segmentation with MobileNetV3 model'
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
    '--orig-size',
    action='store_true',
    help='output in original image size.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = parser.parse_args()


# ======================
# Secondaty Functions
# ======================


def preprocess(img):
    img = img[:, :, ::-1]
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def post_processing(output, orig_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    threshold = 100

    out_mask = np.squeeze(output)
    if args.orig_size:
        out_mask = cv2.resize(
            out_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR
        )
    out_mask = out_mask * 255 > threshold

    return out_mask


# ======================
# Main functions
# ======================


def predict(img, net):
    # initial preprocesses
    h, w, _ = img.shape
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict({
            'input_1': img
        })
    else:
        input_name = net.get_inputs()[0].name
        output_name = net.get_outputs()[0].name
        output = net.run([output_name],
                         {input_name: img})[0]

    # post processes
    out_mask = post_processing(output, orig_shape=(h, w))

    return out_mask


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    print(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            out_mask = predict(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        out_mask = predict(img, net)

    if not args.orig_size:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    res_img = np.ones(img.shape, np.uint8) * 255
    res_img[out_mask] = img[out_mask]

    # plot result
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video(video, net):
    capture = webcamera_utils.get_capture(video)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        out_mask = predict(frame, net)

        # draw segmentation area
        if not args.orig_size:
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        res_img = np.ones(frame.shape, np.uint8) * 255
        res_img[out_mask] = frame[out_mask]

        # show
        cv2.imshow('frame', res_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, net)
    else:
        # image mode
        recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
