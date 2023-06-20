import os
import sys
import time

import numpy as np
import cv2
import transformers

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, urlretrieve, progress_print  # noqa
# logger
from logging import getLogger  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = 'unet.onnx'
WEIGHT_PB_UNET_PATH = 'weights.pb'
MODEL_UNET_PATH = 'unet.onnx.prototxt'
WEIGHT_SAFETY_CHECKER_PATH = 'safety_checker.onnx'
MODEL_SAFETY_CHECKER_PATH = 'safety_checker.onnx.prototxt'
WEIGHT_TEXT_ENCODER_PATH = 'text_encoder.onnx'
MODEL_TEXT_ENCODER_PATH = 'text_encoder.onnx.prototxt'
WEIGHT_VAE_ENCODER_PATH = 'vae_encoder.onnx'
MODEL_VAE_ENCODER_PATH = 'vae_encoder.onnx.prototxt'
WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/anything_v3/'

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Riffusion', None, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="pikachu",
    help="the prompt to render"
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================

def recognize_from_text(pipe):
    # prompt = args.input if isinstance(args.input, str) else args.input[0]
    prompt = "jazzy rapping from paris"
    negative_prompt = ""
    num_clips = 1
    num_inference_steps = 30
    guidance = 7.0
    width = 512
    seed = 42
    logger.info("prompt: %s" % prompt)

    logger.info('Start inference...')

    output = pipe.forward(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt or None,
        width=width,
        height=512,
    )
    image = output.images[0]

    savepath = get_savepath(args.savepath, "", ext='.png')
    logger.info(f'saved at : {savepath}')
    image.save(savepath)

    logger.info('Script finished successfully.')


def main():
    # check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_SAFETY_CHECKER_PATH, MODEL_SAFETY_CHECKER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)

    if not os.path.exists(WEIGHT_PB_UNET_PATH):
        logger.info('Downloading weights.pb...')
        urlretrieve(REMOTE_PATH, WEIGHT_PB_UNET_PATH, progress_print)
    logger.info('weights.pb is prepared!')

    env_id = args.env_id

    # initialize
    if not args.onnx:
        pass
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

        net = onnxruntime.InferenceSession("unet.onnx", providers=providers)
        # vae_decoder = OnnxRuntimeModel.from_pretrained(
        #     "./", "vae_decoder.onnx",
        #     {'provider': 'CPUExecutionProvider', 'sess_options': None}
        # )
        # pndm_scheduler = df.schedulers.scheduling_pndm.PNDMScheduler.from_pretrained(
        #     "./scheduler"
        # )
        # feature_extractor = transformers.CLIPImageProcessor.from_pretrained(
        #     "./feature_extractor"
        # )
        text_encoder = onnxruntime.InferenceSession("text_encoder.onnx", providers=providers)
        # vae_encoder = OnnxRuntimeModel.from_pretrained(
        #     "./", "vae_encoder.onnx",
        #     {'provider': 'CPUExecutionProvider', 'sess_options': None}
        # )
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "./tokenizer"
        )

    pipe = df.StableDiffusion(
        # vae_encoder=vae_encoder,
        # vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=net,
        # scheduler=pndm_scheduler,
        # safety_checker=safety_checker,
        # feature_extractor=feature_extractor,
        # requires_safety_checker=True
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == '__main__':
    main()
