import sys

sys.path.append('../../util')
from utils import get_base_parser, update_parser
from model_utils import check_and_download_models

import numpy as np
import ailia
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================
MODEL_LISTS = [
    'cosface_sphere20_opset10'
]

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/cosface/"
IMG_PATH_1 = 'image_id.jpg'
IMG_PATH_2 = 'image_target.jpg'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Determine if the person is the same from two facial images by CosFace',
    None,
    None,
)

# Overwrite default config
# NOTE: cosface has different usage for `--input` with other models
parser.add_argument(
    '-i', '--inputs', metavar='IMAGE', nargs=2, default='',
    help='Two image paths for calculating the face match.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='CosFace_sphere20_opset10', choices=MODEL_LISTS,
    help='Supported model: ' + ' | '.join(MODEL_LISTS)
)

args = update_parser(parser)

WEIGHT_PATH = args.arch + '.onnx'


def preprocessing(img_path):
    with open(img_path, 'rb') as f:
        original_img = Image.open(f).convert('RGB').resize((96, 112), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    img, img_flipped = transform(original_img), transform(F.hflip(original_img))
    img, img_flipped = img.unsqueeze(0).numpy(), img_flipped.unsqueeze(0).numpy()
    return {'original': img, 'flipped': img_flipped}


def extractDeepFeature(images, model):
    # features of original image and the flipped image are concatenated together
    # to compose the final face representation
    ft = np.concatenate((model.predict(images['original']), model.predict(images['flipped'])), 1)[0]
    return ft


def cosFace(id_img_path, query_img_path):
    # Pre-processing
    img1 = preprocessing(id_img_path)
    img2 = preprocessing(query_img_path)

    # Inference ( extract feature )
    model = ailia.Net(weight=WEIGHT_PATH, env_id=args.env_id)
    f1 = extractDeepFeature(img1, model)
    f2 = extractDeepFeature(img2, model)

    # Post-processing ( cosine distance )
    cosine_distance = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)
    return cosine_distance


def main():
    # Check and download CosFace model
    check_and_download_models(WEIGHT_PATH, None, REMOTE_PATH)

    # Specify default images when there is no inputs
    if len(args.inputs) == 0:
        args.inputs = [IMG_PATH_1, IMG_PATH_2]

    # Comparing two images
    distance = cosFace(args.inputs[0],
                       args.inputs[1])
    logger.info(
        f'Similarity of ({args.inputs[0]}, {args.inputs[1]}) : {distance:.3f}'
    )
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
