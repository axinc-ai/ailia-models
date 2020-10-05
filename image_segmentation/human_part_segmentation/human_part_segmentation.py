import sys
import time
import argparse

from PIL import Image
import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402C
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C

from hps_utils import xywh2cs, transform_logits, get_affine_transform

# ======================
# Parameters
# ======================

WEIGHT_PATH = './resnet-lip.onnx'
MODEL_PATH = './resnet-lip.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/human_part_segmentation/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY = (
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
    'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
    'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'
)
IMAGE_HEIGHT = 473
IMAGE_WIDTH = 473

NORM_MEAN = [0.406, 0.456, 0.485]
NORM_STD = [0.225, 0.224, 0.229]

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Clothing detection model'
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
# Secondaty Functions
# ======================


def preprocess(img):
    h, w, _ = img.shape

    # Get person center and scale
    person_center, s = xywh2cs(0, 0, w - 1, h - 1)
    r = 0
    trans = get_affine_transform(person_center, s, r, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = cv2.warpAffine(
        img,
        trans,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    # normalize
    img = ((img / 255.0 - NORM_MEAN) / NORM_STD).astype(np.float32)

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    data = {
        'img': img,
        'center': person_center,
        'height': h,
        'width': w,
        'scale': s,
        'rotation': r
    }
    return data


def post_processing(data, fusion):
    fusion = fusion[0].transpose(1, 2, 0)
    upsample_output = cv2.resize(fusion, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_LINEAR)
    logits_result = transform_logits(
        upsample_output,
        data['center'], data['scale'], data['width'], data['height'],
        input_size=[IMAGE_HEIGHT, IMAGE_WIDTH]
    )

    pixel_labels = np.argmax(logits_result, axis=2)

    return pixel_labels


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


# ======================
# Main functions
# ======================


def detect_objects(img, detector):
    # initial preprocesses
    data = preprocess(img)

    # feedforward
    output = detector.predict({
        'img': data['img']
    })
    _, fusion, _ = output

    # post processes
    pixel_labels = post_processing(data, fusion)

    return pixel_labels


def recognize_from_image(filename, detector):
    # prepare input data
    img_0 = load_image(filename)
    print(f'input image shape: {img_0.shape}')

    img = cv2.cvtColor(img_0, cv2.COLOR_BGRA2BGR)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pixel_labels = detect_objects(img, detector)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        parsing_result = detect_objects(img, detector)

    output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
    palette = get_palette(len(CATEGORY))
    output_img.putpalette(palette)
    output_img.save(args.savepath)
    print('Script finished successfully.')


def recognize_from_video(video, detector):
    capture = webcamera_utils.get_capture(args.video)

    palette = get_palette(len(CATEGORY))
    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        pixel_labels = detect_objects(frame, detector)

        # draw segmentation area
        mask = pixel_labels != 0
        im = Image.fromarray(np.asarray(pixel_labels, dtype=np.uint8))
        im.putpalette(palette)
        fill = np.asarray(im.convert("RGB"))
        fill = cv2.cvtColor(fill, cv2.COLOR_RGB2BGR)
        frame[mask] = frame[mask] * 0.6 + fill[mask] * 0.4
        # show
        cv2.imshow('frame', frame)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    # initialize
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, detector)
    else:
        # image mode
        recognize_from_image(args.input, detector)


if __name__ == '__main__':
    main()
