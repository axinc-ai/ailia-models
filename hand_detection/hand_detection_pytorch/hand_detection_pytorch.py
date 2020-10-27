import sys
import time
import argparse

import cv2

import ailia
import hand_detection_pytorch_utils

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'hand_detection_pytorch.onnx'
MODEL_PATH = 'hand_detection_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/hand_detection_pytorch/'

IMAGE_PATH = 'CARDS_OFFICE.jpg'
SAVE_IMAGE_PATH = 'CARDS_OFFICE_output.jpg'

THRESHOLD = 0.2
IOU = 0.2


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='hand-detection.PyTorch hand detection model'
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
    to_show = cv2.imread(args.input, cv2.IMREAD_COLOR)
    print(f'input image shape: {to_show.shape}')
    img, scale = hand_detection_pytorch_utils.pre_process(to_show)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    detector.set_input_shape((1, 3, img.shape[2], img.shape[3]))

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            out = detector.predict({'input.1': img})
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        out = detector.predict({'input.1': img})

    dets = hand_detection_pytorch_utils.post_process(out, img, scale, THRESHOLD, IOU)
    for i in range(dets.shape[0]):
        cv2.rectangle(to_show, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)
    cv2.imwrite(args.savepath, to_show)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)
    while(True):
        ret, to_show = capture.read()
        img, scale = hand_detection_pytorch_utils.pre_process(to_show)
        detector.set_input_shape((1, 3, img.shape[2], img.shape[3]))
        out = detector.predict({'input.1': img})
        dets = hand_detection_pytorch_utils.post_process(out, img, scale, THRESHOLD, IOU)
        for i in range(dets.shape[0]):
            cv2.rectangle(to_show, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)
        cv2.imshow('frame', to_show)

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
