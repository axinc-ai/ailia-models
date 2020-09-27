import sys
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402C
from u2net_utils import load_image, transform, save_result, norm  # noqa: E402C


# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 320
MODEL_LISTS = ['small', 'large']
OPSET_LISTS = ['10', '11']


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='U square net'
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
    '-a', '--arch', metavar='ARCH',
    default='large', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
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
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
parser.add_argument(
    '-o', '--opset', metavar='OPSET',
    default='10', choices=OPSET_LISTS,
    help='opset lists: ' + ' | '.join(OPSET_LISTS)
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================
if args.opset=="10":
    WEIGHT_PATH = 'u2net.onnx' if args.arch == 'large' else 'u2netp.onnx'
else:
    WEIGHT_PATH = 'u2net_opset11.onnx' if args.arch == 'large' else 'u2netp_opset11.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    input_data, h, w = load_image(
        args.input,
        scaled_size=IMAGE_SIZE,
    )

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict([input_data])
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        # dim = [(1, 1, 320, 320), (1, 1, 320, 320),..., ]  len=7
        preds_ailia = net.predict([input_data])

    # postprocessing
    # we only use `d1` (the first output, check the original repository)
    pred = preds_ailia[0][0, 0, :, :]

    save_result(pred, args.savepath, [h, w])

    # composite
    if args.composite:
        image = cv2.imread(args.input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:,:,3] = cv2.resize(pred,(w,h)) * 255
        cv2.imwrite(args.savepath, image)

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

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))    
    if args.savepath != SAVE_IMAGE_PATH:    
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w, rgb=False)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_data = transform(frame, IMAGE_SIZE)

        # inference
        preds_ailia = net.predict([input_data])

        # postprocessing
        pred = cv2.resize(norm(preds_ailia[0][0, 0, :, :]), (f_w, f_h))
        cv2.imshow('frame', pred)
        
        # save results
        if writer is not None:
            writer.write((pred * 255).astype(np.uint8))
        
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
