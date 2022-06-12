import sys
import time

import numpy as np
import cv2
from PIL import Image

from transformers import BertTokenizerFast

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# from image_utils import load_image, normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'xxx.onnx'
MODEL_PATH = 'xxx.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Latent Diffusion', None, None
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================


class BERTEmbedder:
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, transformer_emb, transformer_attn, max_length=77):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        self.transformer_emb = transformer_emb
        self.transformer_attn = transformer_attn

    def encode(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]
        tokens = tokens.numpy()

        if not args.onnx:
            output = self.transformer_emb.predict([tokens])
        else:
            output = self.transformer_emb.run(None, {'x': tokens})
        x = output[0]

        if not args.onnx:
            output = self.transformer_attn.predict([x])
        else:
            output = self.transformer_attn.run(None, {'x': x})
        z = output[0]

        return z


def post_processing(output):
    return None


def predict(net, c):
    pred = post_processing(output)

    return pred


def recognize_from_text(
        transformer_emb, transformer_attn,
        diffusion_emb, diffusion_mid, diffusion_out,
        autoencoder):
    n_samples = 4
    n_iter = 1
    scale = 5.0
    cond_stage_model = BERTEmbedder(transformer_emb, transformer_attn)

    uc = None
    if scale != 1.0:
        uc = cond_stage_model.encode([""] * n_samples)

    logger.info('Script finished successfully.')


def main():
    env_id = args.env_id

    # initialize
    if not args.onnx:
        transformer_emb = ailia.Net("transformer_emb.onnx.prototxt", "transformer_emb.onnx", env_id=env_id)
        transformer_attn = ailia.Net("transformer_attn.onnx.prototxt", "transformer_attn.onnx", env_id=env_id)
        diffusion_emb = ailia.Net("diffusion_emb.onnx.prototxt", "diffusion_emb.onnx", env_id=env_id)
        diffusion_mid = ailia.Net("diffusion_mid.onnx.prototxt", "diffusion_mid.onnx", env_id=env_id)
        diffusion_out = ailia.Net("diffusion_out.onnx.prototxt", "diffusion_out.onnx", env_id=env_id)
        autoencoder = ailia.Net("autoencoder.onnx.prototxt", "autoencoder.onnx", env_id=env_id)
    else:
        import onnxruntime
        transformer_emb = onnxruntime.InferenceSession("transformer_emb.onnx", env_id=env_id)
        transformer_attn = onnxruntime.InferenceSession("transformer_attn.onnx", env_id=env_id)
        diffusion_emb = onnxruntime.InferenceSession("diffusion_emb.onnx", env_id=env_id)
        diffusion_mid = onnxruntime.InferenceSession("diffusion_mid.onnx", env_id=env_id)
        diffusion_out = onnxruntime.InferenceSession("diffusion_out.onnx", env_id=env_id)
        autoencoder = onnxruntime.InferenceSession("autoencoder.onnx", env_id=env_id)

    recognize_from_text(
        transformer_emb, transformer_attn,
        diffusion_emb, diffusion_mid, diffusion_out,
        autoencoder)


if __name__ == '__main__':
    main()
