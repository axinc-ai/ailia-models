import sys
import time
import argparse

import cv2
import numpy as np

import ailia
# import original modules
sys.path.append('../util')
from utils import check_file_existance
from model_utils import check_and_download_models
from image_utils import load_image
from webcamera_utils import preprocess_frame


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "crowdcount.onnx"
MODEL_PATH = "crowdcount.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crowd_count/"

IMAGE_PATH = 'test.jpeg'
SAVE_IMAGE_PATH = 'result.png'
WIDTH = 640
HEIGHT = 480


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Single image crowd counting.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
    default=IMAGE_PATH, 
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +\
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
args = parser.parse_args()


# ======================p
# Main functions
# ======================
def estimate_from_image():
    # prepare input data
    org_img = load_image(args.input, (HEIGHT, WIDTH), normalize_type='None')
    img = load_image(
        args.input,
        (HEIGHT, WIDTH),
        rgb=False,
        normalize_type='None',
        gen_input_ailia=True
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(env_id)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # compute execution time
    for i in range(5):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(img)
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end - start))

    # estimated crowd count
    et_count = int(np.sum(preds_ailia))

    # density map
    density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
    density_map = cv2.resize(density_map, (WIDTH, HEIGHT))
    heatmap = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(
        heatmap,
        f'Est Count: {et_count}',
        (40, 440),  # position
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        0.8,  # fontscale
        (255, 255, 255),  # color
        2  # thickness
    )

    res_img = np.hstack((org_img, heatmap))
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def estimate_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(env_id)
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
            frame, HEIGHT, WIDTH, data_rgb=False, normalize_type='None'
        )

        # inference
        preds_ailia = net.predict(input_data)
    
        # estimated crowd count
        et_count = int(np.sum(preds_ailia))

        # density map
        density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
        density_map = cv2.resize(
            density_map,
            (input_image.shape[1], input_image.shape[0])
        )
        heatmap = cv2.applyColorMap(
            density_map.astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.putText(
            heatmap,
            f'Est Count: {et_count}',
            (40, 440),  # position
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            0.8,  # fontscale
            (255, 255, 255),  # color
            2  # thickness
        )
        res_img = np.hstack((input_image, heatmap))
        cv2.imshow('frame', res_img)
    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video:
        # video mode
        estimate_from_video()
    else:
        # image mode
        estimate_from_image()


if __name__ == "__main__":
    main()
