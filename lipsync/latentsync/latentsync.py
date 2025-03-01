import sys

import numpy as np

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
WEIGHT_VAE_DEC_PATH = "vae_decoder.onnx"
MODEL_VAE_DEC_PATH = "vae_decoder.onnx.prototxt"
WEIGHT_VAE_ENC_PATH = "vae_encoder.onnx"
MODEL_VAE_ENC_PATH = "vae_encoder.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/latentsync/"

IMAGE_SIZE = 512
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync",
    None,
    SAVE_IMAGE_PATH,
    fp16_support=False,
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================


def recognize_offline(pipe: df.LipsyncPipeline):
    logger.info("Start inference...")

    pipe.forward(
        num_frames=16,
        num_inference_steps=20,
        guidance_scale=1.5,
        height=256,
        width=256,
    )

    # img_savepath = get_savepath(args.savepath, "", ext=".png")
    # logger.info(f"saved at : {img_savepath}")
    # cv2.imwrite(img_savepath, image)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_ENC_PATH, MODEL_VAE_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VAE_DEC_PATH, MODEL_VAE_DEC_PATH, REMOTE_PATH)

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
        vae_encoder = ailia.Net(
            MODEL_VAE_ENC_PATH,
            WEIGHT_VAE_ENC_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        net = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        vae_encoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_ENC_PATH, providers=providers
        )

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = None
    else:
        raise NotImplementedError

    # scheduler = df.schedulers.EulerAncestralDiscreteScheduler.from_config(
    #     {
    #         "num_train_timesteps": 1000,
    #         "timestep_spacing": "trailing",
    #         "beta_start": 0.00085,
    #         "beta_end": 0.012,
    #         "beta_schedule": "scaled_linear",
    #         "trained_betas": None,
    #         "steps_offset": 1,
    #         "prediction_type": "epsilon",
    #     }
    # )
    scheduler = None

    pipe = df.LipsyncPipeline(
        vae_encoder=vae_encoder,
        tokenizer=tokenizer,
        unet=net,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_offline(pipe)


if __name__ == "__main__":
    main()
