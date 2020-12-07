import sys
import time
import argparse

import cv2

import ailia
import craft_pytorch_utils

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'craft.onnx'
MODEL_PATH = 'craft.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/craft-pytorch/'

IMAGE_PATH = 'imgs/00_00.jpg'
SAVE_IMAGE_PATH = 'imgs_results/res_00_00.jpg'

THRESHOLD = 0.2
IOU = 0.2


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='CRAFT: Character-Region Awareness For Text detection'
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
args = parser.parse_args()


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    image = craft_pytorch_utils.load_image(args.input)
    print(f'input image shape: {image.shape}')
    x, ratio_w, ratio_h = craft_pytorch_utils.pre_process(image)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, 3, x.shape[2], x.shape[3]))

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            y, _ = net.predict({'input.1': x})
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        y, _ = net.predict({'input.1': x})

    img = craft_pytorch_utils.post_process(y, image, ratio_w, ratio_h)
    cv2.imwrite(args.savepath, img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)
    while(True):
        ret, image = capture.read()
        x, ratio_w, ratio_h = craft_pytorch_utils.pre_process(image)
        net.set_input_shape((1, 3, x.shape[2], x.shape[3]))
        y, _ = net.predict({'input.1': x})
        img = craft_pytorch_utils.post_process(y, image, ratio_w, ratio_h)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', img)

        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

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
