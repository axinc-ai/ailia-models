import sys, os
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'crnn.onnx'
MODEL_PATH = 'crnn.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/crnn_pytorch/'

IMAGE_PATH = 'demo.png'
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 100

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Convolutional Recurrent Neural Network'
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
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255
    img = img - 0.5
    img = img / 0.5
    img = img.reshape(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    return img


def post_processing(preds):
    preds = np.argmax(preds, axis=2)
    preds = preds.transpose(1, 0)
    preds = preds.reshape(-1)
    return preds


def decode(t, raw=False):
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz' + '-'
    dic = {char: i + 1 for i, char in enumerate(alphabet)}

    if raw:
        return ''.join([alphabet[i - 1] for i in t])
    else:
        char_list = []
        for i in range(len(t)):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(alphabet[t[i] - 1])
        return ''.join(char_list)


# ======================
# Main functions
# ======================


def predict(img, net):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        preds = net.predict({'img': img})[0]
    else:
        in_img = net.get_inputs()[0].name
        out_preds = net.get_outputs()[0].name
        preds = net.run([out_preds],
                        {in_img: img})[0]

    preds = post_processing(preds)

    return preds


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    print(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds = predict(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds = predict(img, net)

    raw_pred = decode(preds, raw=True)
    sim_pred = decode(preds, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))


def recognize_from_video(video, net):
    capture = get_capture(video)

    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preds = predict(img, net)

        # plot result
        sim_pred = decode(preds, raw=False)
        cv2.putText(
            frame, sim_pred, (1, 20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1, cv2.LINE_AA)

        # show
        cv2.imshow('frame', frame)

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
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(args.video, net)
    else:
        recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
