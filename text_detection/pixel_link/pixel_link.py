import sys, os
import time
import argparse

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from pixel_link_utils import decode_batch, mask_to_bboxes, draw_bbox

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'pixellink-vgg16-4s.onnx'
MODEL_PATH = 'pixellink-vgg16-4s.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pixel_link/'

IMAGE_PATH = 'img_249.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Pixel-Link model'
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
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH', default=SAVE_IMAGE_PATH,
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
# Secondaty Functions
# ======================


def post_processing(pixel_pos_scores, link_pos_scores, image_shape):
    mask = decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    bboxes = mask_to_bboxes(mask, image_shape)

    return bboxes


# ======================
# Main functions
# ======================


def predict(img, net):
    img = img.astype(np.int32)

    # feedforward
    net.set_input_shape(img.shape)
    output = net.predict({'import/test/Placeholder:0': img})

    pixel_pos_scores, link_pos_scores = output
    bboxes = post_processing(pixel_pos_scores, link_pos_scores, img.shape)

    return bboxes


def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    print(f'input image shape: {img.shape}')

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            bboxes = predict(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        bboxes = predict(img, net)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res_img = draw_bbox(img, bboxes)

    # plot result
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video(video, net):
    capture = get_capture(video)

    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        bboxes = predict(frame, net)

        # plot result
        res_img = draw_bbox(frame, bboxes)

        # show
        cv2.imshow('frame', res_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    memory_mode=ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
    print(f'env_id: {env_id}')

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)

    if args.video is not None:
        recognize_from_video(args.video, net)
    else:
        recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
