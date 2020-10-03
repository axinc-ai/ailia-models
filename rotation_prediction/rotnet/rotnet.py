import sys
import time
import argparse

import cv2
import numpy as np

import ailia
from rotnet_utils import generate_rotated_image, create_figure, visualize

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C


# ======================
# Parameters 1
# ======================
MODEL_NAMES = ['mnist', 'gsv2']
MODEL_DICT = {
    'mnist': "rotnet_mnist",
    'gsv2': "rotnet_gsv_2"
}
IMAGE_PATH = 'test.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Image Rotation Correction Model'
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
    '--model', '-m', metavar='model',
    default='gsv2', choices=MODEL_NAMES,
    help=('model architecture: ' + ' | '.join(MODEL_NAMES) +
          ' (default: gsv2)')
)
parser.add_argument(
    '--apply_rotate', action='store_true',
    help='If add this argument, apply random rotation to input image'
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
WEIGHT_PATH = MODEL_DICT[args.model] + '.onnx'
MODEL_PATH = MODEL_DICT[args.model] + '.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/rotnet/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)

    if args.apply_rotate:
        rotation_angle = np.random.randint(360)
        rotated_img = generate_rotated_image(
            org_img,
            rotation_angle,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            crop_center=True,
            crop_largest_rect=True
        )
        input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    else:
        rotation_angle = 0
        rotated_img = cv2.resize(org_img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape(input_data.shape)

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

    # visualize
    predicted_angle = np.argmax(preds_ailia, axis=1)[0]
    fig = create_figure()
    plt = visualize(fig, rotated_img, rotation_angle, predicted_angle)
    plt.savefig(args.savepath)

    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    net.set_input_shape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    capture = get_capture(args.video)

    fig = create_figure()
    tight_layout = True

    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        input_image, resized_img = adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        if args.apply_rotate:
            rotation_angle = np.random.randint(360)
            rotated_img = generate_rotated_image(
                resized_img,
                rotation_angle,
                size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                crop_center=True,
                crop_largest_rect=True
            )
            input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        else:
            rotation_angle = 0
            rotated_img = resized_img
            input_data = rotated_img.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        # inference
        preds_ailia = net.predict(input_data)

        # visualize
        predicted_angle = np.argmax(preds_ailia, axis=1)[0]
        plt = visualize(
            fig, rotated_img, rotation_angle, predicted_angle, tight_layout
        )
        plt.pause(.01)
        tight_layout = False

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
