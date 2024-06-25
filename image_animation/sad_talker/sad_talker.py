import sys
from logging import getLogger

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser
from image_animation.sad_talker.image_processor import ImageToCoeff

logger = getLogger(__name__)

INPUT_IMAGE_PATH    = "./sample/input.png"
INPUT_AUDIO_PATH    = "./sample/input.wav"
OUTPUT_PATH         = "./result.mp4"

# NOTE: Do not use default input argument due to the default parser is not compatible with multiple inputs.
parser = get_base_parser(
    'Sad Talker', None, OUTPUT_PATH, None,
)
parser.add_argument(
    '--image',
    type=str,
    default=INPUT_IMAGE_PATH,
    metavar="IMAGE",
    help='path to input image file (png, jpg)',
)
parser.add_argument(
    '--audio',
    type=str,
    default=INPUT_AUDIO_PATH,
    metavar="AUDIO",
    help='path to input audio file (wav)',
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '--preprocess',
    type=str,
    choices=["resize", "crop"],
    help='execute onnxruntime version.'
)
args = update_parser(parser)


def main():
    preprocesor = ImageToCoeff(
        use_onnx=True,
    )
    print('3DMM Extraction for source image')



if __name__ == '__main__':
    main()
