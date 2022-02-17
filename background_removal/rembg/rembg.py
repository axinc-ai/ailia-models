import sys
import time

import numpy as np
import cv2
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from scipy.ndimage.morphology import binary_erosion

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image  # noqa
# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

MASK_WEIGHT_PATH = 'u2net_opset11.onnx'
MASK_MODEL_PATH = MASK_WEIGHT_PATH + '.prototxt'
MASK_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net/'

IMAGE_PATH = 'animal-1.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser('Rembg', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def norm_pred(d: np.ndarray) -> np.ndarray:
    ma = np.max(d)
    mi = np.min(d)
    return (d - mi) / (ma - mi)


# ======================
# Main functions
# ======================

def preprocess(img):
    # img = img[:, :, ::-1]  # BGR -> RGB

    from skimage import transform
    img = transform.resize(img, (320, 320), mode="constant")
    img = img / np.max(img)

    # normalize
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def estimate_alpha(
        img, mask,
        foreground_threshold=240,
        background_threshold=10,
        erode_structure_size=10):
    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    return alpha


def predict(net, img):
    im_h, im_w = img.shape[:2]

    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    d1 = output[0]

    pred = d1[0, 0, :, :]
    pred = norm_pred(pred)

    # pred = cv2.resize(pred * 255, (w, h))
    # pred = pred.reshape((h, w, 1))
    mask = Image.fromarray(pred * 255).convert("L")
    mask = mask.resize((im_w, im_h), Image.LANCZOS)
    mask = np.array(mask)

    return mask


def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        # img = load_image(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        # inference
        mask = predict(net, img)
        res_img = estimate_alpha(img, mask)

        if args.composite:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
            img[:, :, 3] = res_img
            res_img = img

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(MASK_WEIGHT_PATH, MASK_MODEL_PATH, MASK_REMOTE_PATH)

    # load model
    env_id = args.env_id

    # net initialize
    net = ailia.Net(MASK_MODEL_PATH, MASK_WEIGHT_PATH, env_id=env_id)

    recognize_from_image(net)


if __name__ == '__main__':
    main()
