import time
import sys

import numpy as np

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import *  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402 logger = getLogger(__name__)

from inaSpeechSegmenter_util import *

# ======================
# Arguemnt Parser Config
# ======================
AUDIO_PATH = 'musanmix.mp3'
parser = get_base_parser('ina speech segmenter', AUDIO_PATH, None)
parser.add_argument(
    '-m','--model',
    type=str,
    default='smn',
    help='[smn,sm]'
)
parser.add_argument(
    '--gender',
    action='store_false'
)
args = update_parser(parser)


# ======================
# PARAMETERS
# ======================
SM_AUDIO_WEIGHT_PATH = "inaSpeechSegmenter_sm.onnx"
SM_AUDIO_MODEL_PATH  = "inaSpeechSegmenter_sm.onnx.prototxt"

SMN_AUDIO_WEIGHT_PATH = "inaSpeechSegmenter_smn.onnx"
SMN_AUDIO_MODEL_PATH  = "inaSpeechSegmenter_smn.onnx.prototxt"

GENDER_AUDIO_WEIGHT_PATH = "inaSpeechSegmenter_gender.onnx"
GENDER_AUDIO_MODEL_PATH  = "inaSpeechSegmenter_gender.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/ina_speech_segmenter/"



# ======================
# Main function
# ======================
def infer_audio(audio_path):
    # tokenizer
    seg = Segmenter(vad_engine=args.model)
    
    # Run the segmentation
    segmentation = seg(audio_path)
    for i in range(len(segmentation)):
        print("labels:" ,segmentation[i][0],",start:",segmentation[i][1],",stop:",segmentation[i][2])

def main():
    # model files check and download
    check_and_download_models(SM_AUDIO_WEIGHT_PATH, SM_AUDIO_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(SMN_AUDIO_WEIGHT_PATH, SMN_AUDIO_MODEL_PATH, REMOTE_PATH)
    check_and_download_models(GENDER_AUDIO_WEIGHT_PATH, GENDER_AUDIO_MODEL_PATH, REMOTE_PATH)

    # audio predict
    for audio_path in args.input:
        audio_embedding = infer_audio(audio_path)

    logger.info('Script finished successfully.')


if __name__ == "__main__":
    main()
