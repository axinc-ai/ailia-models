import sys
import time
import argparse

import PIL.Image as pil
import cv2
import matplotlib.pyplot as plt
import numpy as np

import ailia

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402C
from webcamera_utils import preprocess_frame  # noqa: E402C


# ======================
# Parameters
# ======================
MODEL_NAME = 'monodepth2_mono+stereo_640x192'
ENC_WEIGHT_PATH = MODEL_NAME + '_enc.onnx'
ENC_MODEL_PATH = MODEL_NAME + '_enc.onnx.prototxt'
DEC_WEIGHT_PATH = MODEL_NAME + '_dec.onnx'
DEC_MODEL_PATH = MODEL_NAME + '_dec.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/monodepth2/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Depth estimation model'
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
args = parser.parse_args()


# ======================
# Utils
# ======================
def result_plot(disp, original_width, original_height):
    disp = disp.squeeze()
    disp_resized = cv2.resize(
        disp,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    vmax = np.percentile(disp_resized, 95)
    return disp_resized, vmax


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_height, org_width, _ = cv2.imread(args.input).shape
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        gen_input_ailia=True
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    enc_net = ailia.Net(ENC_MODEL_PATH, ENC_WEIGHT_PATH, env_id=env_id)
    dec_net = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=env_id)

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))

        # encoder
        enc_input_blobs = enc_net.get_input_blob_list()
        enc_net.set_input_blob_data(input_data, enc_input_blobs[0])
        enc_net.update()
        features = enc_net.get_results()

        # decoder
        dec_inputs_blobs = dec_net.get_input_blob_list()
        for f_idx in range(len(features)):
            dec_net.set_input_blob_data(
                features[f_idx], dec_inputs_blobs[f_idx]
            )
        dec_net.update()
        preds_ailia = dec_net.get_results()

        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')

    # postprocessing
    disp = preds_ailia[-1]
    disp_resized, vmax = result_plot(disp, org_width, org_height)
    plt.imsave(args.savepath, disp_resized, cmap='magma', vmax=vmax)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    enc_net = ailia.Net(ENC_MODEL_PATH, ENC_WEIGHT_PATH, env_id=env_id)
    dec_net = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=env_id)

    if args.video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if check_file_existance(args.video):
            capture = cv2.VideoCapture(args.video)

    ret, frame = capture.read()
    org_height, org_width, _ = frame.shape
    
    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        _, input_data = preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )

        # encoder
        enc_input_blobs = enc_net.get_input_blob_list()
        enc_net.set_input_blob_data(input_data, enc_input_blobs[0])
        enc_net.update()
        features = enc_net.get_results()

        # decoder
        dec_inputs_blobs = dec_net.get_input_blob_list()
        for f_idx in range(len(features)):
            dec_net.set_input_blob_data(
                features[f_idx], dec_inputs_blobs[f_idx]
            )
        dec_net.update()
        preds_ailia = dec_net.get_results()

        # postprocessing
        disp = preds_ailia[-1]
        disp_resized, vmax =  result_plot(disp, org_width, org_height)
        plt.imshow(disp_resized, cmap='magma', vmax=vmax)
        plt.pause(.01)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(ENC_WEIGHT_PATH, ENC_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(DEC_WEIGHT_PATH, DEC_MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
