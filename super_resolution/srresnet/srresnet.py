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
from webcamera_utils import preprocess_frame, get_capture  # noqa: E402


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'lenna.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 64    # net.get_input_shape()[3]
IMAGE_WIDTH = 64     # net.get_input_shape()[2]
OUTPUT_HEIGHT = 256  # net.get_output_shape()[3]
OUTPUT_WIDTH = 256   # net.get_output.shape()[2]


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Single Image Super-Resolution'
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
    '-n', '--normal', action='store_true',
    help=('By default, the optimized model is used, but with this option, ' +
          'you can switch to the normal (not optimized) model')
)
parser.add_argument(
    '-p', '--padding', action='store_true',
    help=('Instead of resizing input image when loading it, ' +
          ' padding input and output image')
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
if not args.normal:
    WEIGHT_PATH = 'srresnet.opt.onnx'
    MODEL_PATH = 'srresnet.opt.onnx.prototxt'
else:
    WEIGHT_PATH = 'srresnet.onnx'
    MODEL_PATH = 'srresnet.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/srresnet/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='255',
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
    output_img = preds_ailia[0].transpose((1, 2, 0))
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.savepath, output_img * 255)
    print('Script finished successfully.')


def tiling(net, img):
    h, w = img.shape[0], img.shape[1]

    padding_w = int((w + IMAGE_WIDTH - 1) / IMAGE_WIDTH) * IMAGE_WIDTH
    padding_h = int((h+IMAGE_HEIGHT-1) / IMAGE_HEIGHT) * IMAGE_HEIGHT
    scale = int(OUTPUT_HEIGHT / IMAGE_HEIGHT)
    output_padding_w = padding_w * scale
    output_padding_h = padding_h * scale
    output_w = w * scale
    output_h = h * scale

    print(f'input image : {h}x{w}')
    print(f'output image : {output_w}x{output_h}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]

    pad_img = np.zeros((1, 3, padding_h, padding_w))
    pad_img[:, :, 0:h, 0:w] = img

    output_pad_img = np.zeros((1, 3, output_padding_h, output_padding_w))
    tile_x = int(padding_w / IMAGE_WIDTH)
    tile_y = int(padding_h / IMAGE_HEIGHT)

    # Inference
    start = int(round(time.time() * 1000))
    for y in range(tile_y):
        for x in range(tile_x):
            output_pad_img[
                :,
                :,
                y*OUTPUT_HEIGHT:(y+1)*OUTPUT_HEIGHT,
                x*OUTPUT_WIDTH:(x+1)*OUTPUT_WIDTH
            ] = net.predict(pad_img[
                :,
                :,
                y*IMAGE_HEIGHT:(y+1)*IMAGE_HEIGHT,
                x*IMAGE_WIDTH:(x+1)*IMAGE_WIDTH
            ])
    end = int(round(time.time() * 1000))
    print(f'ailia processing time {end - start} ms')

    # Postprocessing
    output_img = output_pad_img[0, :, :output_h, :output_w]
    output_img = output_img.transpose(1, 2, 0).astype(np.float32)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return output_img


def recognize_from_image_tiling():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # processing
    img = cv2.imread(args.input)
    output_img = tiling(net, img)

    cv2.imwrite(args.savepath, output_img * 255)
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
        
        h, w = frame.shape[0], frame.shape[1]
        frame = frame[h//2:h//2+h//4,w//2:w//2+w//4,:]

        output_img = tiling(net, frame)

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
        if args.padding:
            recognize_from_image_tiling()
        else:
            recognize_from_image()


if __name__ == '__main__':
    main()
