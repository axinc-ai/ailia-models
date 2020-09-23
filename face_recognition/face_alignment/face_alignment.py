import sys
import time
import argparse

from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import matplotlib.pyplot as plt

import ailia
# import original modules
sys.path.append('../../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402


# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'face_alignment.onnx.prototxt'
WEIGHT_PATH = 'face_alignment.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face_alignment/'

IMAGE_PATH = 'aflw-test.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
THRESHOLD = 0.1


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Face alignment model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
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
# Utils
# ======================
def plot_images(title, images, tile_shape):
    fig = plt.figure()
    plt.title(title)
    grid = ImageGrid(fig, 111,  nrows_ncols=tile_shape, share_all=True)

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for i in range(images.shape[0]):
        grd = grid[i]
        grd.imshow(images[i])
    split_fname = args.savepath.split('.')
    save_name = split_fname[0] + '_confidence.' + split_fname[1]
    fig.savefig(save_name)


def visualize_plots(image, preds_ailia):
    for i in range(preds_ailia.shape[0]):
        probMap = preds_ailia[i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (image.shape[1] * point[0]) / preds_ailia.shape[2]
        y = (image.shape[0] * point[1]) / preds_ailia.shape[1]

        if prob > THRESHOLD:
            circle_size = 4
            cv2.circle(
                image,
                (int(x), int(y)),
                circle_size,
                (0, 255, 255),
                thickness=-1,
                lineType=cv2.FILLED
            )
    return image


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    input_img = cv2.imread(args.input)
    data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True
    )

    # net initalize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(data)[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(data)[0]

    visualize_plots(input_img, preds_ailia)
    cv2.imwrite(args.savepath, input_img)

    # Confidence Map
    channels = preds_ailia.shape[0]
    cols = 8
    plot_images(
        'confidence',
        preds_ailia,
        tile_shape=((int)((channels+cols-1)/cols), cols))
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
        preds_ailia = net.predict(input_data)[0]

        # postprocessing
        visualize_plots(input_image, preds_ailia)
        cv2.imshow('frame', input_image)

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
