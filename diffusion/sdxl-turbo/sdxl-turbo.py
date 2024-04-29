import os
import sys

import numpy as np
import cv2

import transformers

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

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
WEIGHT_TEXT_ENCODER_2_PATH = "text_encoder_2.onnx"
WEIGHT_TEXT_ENCODER_2_PB_PATH = "text_encoder_2_weights.pb"
MODEL_TEXT_ENCODER_2_PATH = "text_encoder_2.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
MODEL_VAE_DECODER_PATH = "vae_decoder.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sdlx-turbo/"

SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SDXL-Turbo", None, SAVE_IMAGE_PATH)
parser.add_argument(
    "-i",
    "--input",
    metavar="TEXT",
    type=str,
    default="little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed",
    help="the prompt to render",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


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

    img_savepath = get_savepath(args.savepath, "", ext=".png")
    logger.info(f"saved at : {img_savepath}")
    cv2.imwrite(img_savepath, image)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_TEXT_ENCODER_2_PATH, MODEL_TEXT_ENCODER_2_PATH, REMOTE_PATH
    )
    check_and_download_models(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH
    )
    check_and_download_file(WEIGHT_UNET_PB_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_TEXT_ENCODER_2_PB_PATH, REMOTE_PATH)

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
        text_encoder_2 = ailia.Net(
            MODEL_TEXT_ENCODER_2_PATH,
            WEIGHT_TEXT_ENCODER_2_PATH,
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

        net = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_PATH, providers=providers
        )
        text_encoder_2 = onnxruntime.InferenceSession(
            WEIGHT_TEXT_ENCODER_2_PATH, providers=providers
        )
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DECODER_PATH, providers=providers
        )

    tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
    tokenizer_2 = transformers.CLIPTokenizer.from_pretrained("./tokenizer_2")
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
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=net,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == "__main__":
    main()
