import sys
import time
import argparse

import cv2

import ailia
from blazeface_utils import *

# import original modules
sys.path.append('../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from webcamera_utils import preprocess_frame  # noqa: E402


# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = 'blazeface.onnx'
MODEL_PATH = 'blazeface.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'result.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='BlazeFace is a fast and light-weight face detector.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
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
    org_img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='127.5',
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
            preds_ailia = net.predict([input_data])
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict([input_data])

    # postprocessing
    detections = postprocess(preds_ailia)

    # generate detections
    for detection in detections:
        plot_detections(org_img, detection, save_image_path=args.savepath)
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
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5'
        )
        
        # inference
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(input_data, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()

        # postprocessing
        detections = postprocess(preds_ailia)
        show_result(input_image, detections)
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
