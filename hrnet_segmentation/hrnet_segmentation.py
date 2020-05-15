import sys
import time
import argparse

import matplotlib.pyplot as plt
import cv2

import ailia
from hrnet_utils import save_pred, gen_preds_img, smooth_output

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402


# ======================
# Parameters 1
# ======================
MODEL_NAMES = ['HRNetV2-W48', 'HRNetV2-W18-Small-v1', 'HRNetV2-W18-Small-v2']
IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'result.png'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='High-Resolution networks for semantic segmentations.'
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
    '-a', '--arch', metavar="ARCH",
    default='HRNetV2-W18-Small-v2',
    choices=MODEL_NAMES,
    help='model architecture:  ' + ' | '.join(MODEL_NAMES) +
         ' (default: HRNetV2-W18-Small-v2)'
)
parser.add_argument(
    '--smooth',  # '-s' has already been reserved for '--savepath'
    action='store_true',
    help='result image will be smoother by applying bilinear upsampling'
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
MODEL_NAME = args.arch
WEIGHT_PATH = MODEL_NAME + ".onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hrnet/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        gen_input_ailia=True
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute execution time
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
    if args.smooth:
        preds_ailia = smooth_output(preds_ailia)

    save_pred(preds_ailia, args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH)
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
            frame, IMAGE_HEIGHT, IMAGE_WIDTH,
        )

        # inference
        preds_ailia = net.predict(input_data)

        # postprocessing
        if args.smooth:
            preds_ailia = smooth_output(preds_ailia)
        gen_img = gen_preds_img(preds_ailia, IMAGE_HEIGHT, IMAGE_WIDTH)
        plt.imshow(gen_img)
        plt.pause(.01)

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
