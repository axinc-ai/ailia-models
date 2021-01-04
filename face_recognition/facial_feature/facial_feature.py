import os
import sys
import time

from matplotlib import pyplot as plt

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
sys.path.append('../../face_detection/blazeface')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402


# TODO Upgrade Model
# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'resnet_facial_feature.onnx'
MODEL_PATH = 'resnet_facial_feature.onnx.prototxt'
REMOTE_PATH = \
    "https://storage.googleapis.com/ailia-models/resnet_facial_feature/"

IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 226
IMAGE_WIDTH = 226

FACE_WEIGHT_PATH = 'blazeface.onnx'
FACE_MODEL_PATH = 'blazeface.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_MARGIN = 1.0


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'kaggle facial keypoints.', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def gen_img_from_predsailia(input_data, preds_ailia):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(input_data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
    points = np.vstack(np.split(preds_ailia, 15)).T * 113 + 113
    ax.plot(points[0], points[1], 'o', color='red')
    return fig


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        gen_input_ailia=True
    )

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(img)[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(img)[0]

    # postprocess
    fig = gen_img_from_predsailia(img, preds_ailia)
    fig.savefig(args.savepath)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=args.env_id)

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # detect face
        detections = compute_blazeface(
            detector,
            frame,
            anchor_path='../../face_detection/blazeface/anchors.npy',
        )

        # get detected face
        if len(detections) == 0:
            crop_img = frame
        else:
            crop_img, top_left, bottom_right = crop_blazeface(
                detections[0], FACE_MARGIN, frame
            )
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                crop_img = frame

        # preprocess
        input_image, input_data = webcamera_utils.preprocess_frame(
            crop_img, IMAGE_HEIGHT, IMAGE_WIDTH, data_rgb=False
        )

        # inference
        preds_ailia = net.predict(input_data)[0]

        # postprocessing
        fig = gen_img_from_predsailia(input_data, preds_ailia)
        fig.savefig('tmp.png')
        img = cv2.imread('tmp.png')
        cv2.imshow('frame', img)

        # save results
        if writer is not None:
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    os.remove('tmp.png')
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        check_and_download_models(
            FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
        )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
