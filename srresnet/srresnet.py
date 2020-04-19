import sys
import time
import argparse

import cv2

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402C


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'lenna.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64


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

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(input_data)
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    output_img = preds_ailia[0].transpose((1, 2, 0))
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.savepath, output_img * 255)
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

        input_image, input_data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255'
        )

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        output_img = preds_ailia[0].transpose((1, 2, 0))
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
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
