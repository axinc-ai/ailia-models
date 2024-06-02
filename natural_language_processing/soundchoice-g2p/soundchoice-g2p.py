import sys
import time
import re
from functools import partial
from logging import getLogger

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# from sample_utils import decode_batch, mask_to_bboxes, draw_bbox

# ======================
# Parameters
# ======================

WEIGHT_PATH = "soundchoice-g2p.onnx"
MODEL_PATH = "soundchoice-g2p.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/soundchoice-g2p/"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SoundChoice", None, None)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="To be or not to be, that is the question",
    help="Input text.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


def clean_pipeline(txt, graphemes):
    """
    Cleans incoming text, removing any characters not on the
    accepted list of graphemes and converting to uppercase

    Arguments
    ---------
    txt: str
        the text to clean up
    graphemes: list
        a list of graphemes

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    RE_MULTI_SPACE = re.compile(r"\s{2,}")

    result = txt.upper()
    result = "".join(char for char in result if char in graphemes)
    result = RE_MULTI_SPACE.sub(" ", result)
    return result


def grapheme_pipeline(char, uppercase=True):
    """Encodes a grapheme sequence

    Arguments
    ---------
    graphemes: list
        a list of available graphemes
    uppercase: bool
        whether or not to convert items to uppercase

    Returns
    -------
    grapheme_list: list
        a raw list of graphemes, excluding any non-matching
        labels
    grapheme_encoded_list: list
        a list of graphemes encoded as integers
    grapheme_encoded: torch.Tensor
    """
    lab2ind = {
        # fmt: off
        '<bos>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28, "'": 29, ' ': 30
        # fmt: on
    }
    if uppercase:
        char = char.upper()
    grapheme_list = [
        # grapheme for grapheme in char if grapheme in grapheme_encoder.lab2ind
        grapheme
        for grapheme in char
        if grapheme in lab2ind
    ]

    def encode_label(label):
        """Encode label to int

        Arguments
        ---------
        label : hashable
            Label to encode, must exist in the mapping.
        Returns
        -------
        int
            Corresponding encoded int value.
        """
        try:
            return lab2ind[label]
        except KeyError:
            unk_label = "<unk>"
            return lab2ind[unk_label]

    grapheme_encoded_list = [encode_label(label) for label in grapheme_list]


# ======================
# Main functions
# ======================


def encode_input(data):
    intermediate = {}

    graphemes = [
        # fmt: off
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' '
        # fmt: on
    ]
    partial_clean_pipeline = partial(clean_pipeline, graphemes=graphemes)
    value = partial_clean_pipeline(data["txt"])
    intermediate["txt_cleaned"] = value

    value = grapheme_pipeline(value)
    intermediate["grapheme_encoded_list"] = value


def predict(net, input_text):
    model_input = encode_input({"txt": input_text})

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {"src": img})

    return


def recognize_from_text(net):
    input_text = args.input

    logger.info("Input text: " + input_text)

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            out = predict(net, input_text)
            end = int(round(time.time() * 1000))
            estimation_time = end - start

            # Logging
            logger.info(f"\tailia processing estimation time {estimation_time} ms")
            if i != 0:
                total_time_estimation = total_time_estimation + estimation_time

        logger.info(
            f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
        )
    else:
        out = predict(net, input_text)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_text(net)


if __name__ == "__main__":
    main()
