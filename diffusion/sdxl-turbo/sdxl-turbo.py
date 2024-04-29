import os
import sys
import io

import numpy as np
import cv2

import torch
import torchaudio
import transformers

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, urlretrieve, progress_print  # noqa

# logger
from logging import getLogger  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = "unet.onnx"
WEIGHT_UNET_PB_PATH = "unet_weights.pb"
MODEL_UNET_PATH = "unet.onnx.prototxt"
WEIGHT_TEXT_ENCODER_PATH = "text_encoder.onnx"
MODEL_TEXT_ENCODER_PATH = "text_encoder.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "vae_decoder.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sdlx-turbo/"

SAVE_WAV_PATH = "output.wav"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Riffusion", None, SAVE_WAV_PATH)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="jazzy rapping from paris",
    help="the prompt to render",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed",
)
parser.add_argument(
    "--width",
    type=int,
    default=512,
    help="width",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================


# ======================
# Main functions
# ======================


def recognize_from_text(pipe):
    prompt = args.input if isinstance(args.input, str) else args.input[0]
    logger.info("prompt: %s" % prompt)

    logger.info("Start inference...")

    image = pipe.forward(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
    )
    image = (image[0] * 255).astype(np.uint8)

    audio_savepath = get_savepath(args.savepath, "", ext=".wav")
    p, _ = os.path.splitext(audio_savepath)
    img_savepath = p + ".png"
    logger.info(f"saved at : {img_savepath}")
    cv2.imwrite(img_savepath, image)

    logger.info("Script finished successfully.")


def main():
    # check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    # check_and_download_models(
    #     WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    # )
    # check_and_download_models(
    #     WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    # )

    # if not os.path.exists(WEIGHT_UNET_PB_PATH):
    #     urlretrieve(
    #         REMOTE_PATH + WEIGHT_UNET_PB_PATH,
    #         WEIGHT_UNET_PB_PATH,
    #         progress_print,
    #     )

    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        net = ailia.Net(
            MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=env_id, memory_mode=memory_mode
        )
        text_encoder = ailia.Net(
            MODEL_TEXT_ENCODER_PATH,
            WEIGHT_TEXT_ENCODER_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        vae_decoder = ailia.Net(
            MODEL_VAE_DECODER_PATH,
            WEIGHT_VAE_DECODER_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
    else:
        import onnxruntime

        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )

        # net = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        # text_encoder = onnxruntime.InferenceSession(
        #     WEIGHT_TEXT_ENCODER_PATH, providers=providers
        # )
        # vae_decoder = onnxruntime.InferenceSession(
        #     WEIGHT_VAE_DECODER_PATH, providers=providers
        # )

    # tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
    scheduler = df.schedulers.EulerAncestralDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "timestep_spacing": "trailing",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "trained_betas": None,
            "steps_offset": 1,
            "prediction_type": "epsilon",
        }
    )

    pipe = df.StableDiffusionXL(
        vae_decoder=vae_decoder,
        # text_encoder=text_encoder,
        # tokenizer=tokenizer,
        unet=net,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == "__main__":
    main()
