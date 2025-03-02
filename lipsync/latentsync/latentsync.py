import sys
from logging import getLogger

import cv2
import numpy as np
import soundfile as sf
import tqdm

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

import df
from image_processor import AlignRestore

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
# Secondary Functions
# ======================


def write_video(video_output_path: str, video_frames: np.ndarray, fps: int):
    height, width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(
        video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in video_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def restore_video(faces, video_frames, boxes, affine_matrices):
    import torch
    import torchvision
    from einops import rearrange

    logger.info(f"Restoring {len(faces)} faces...")

    restorer = AlignRestore()
    video_frames = video_frames[: faces.shape[0]]

    out_frames = []
    for index, face in enumerate(tqdm.tqdm(faces)):
        x1, y1, x2, y2 = boxes[index]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face = torchvision.transforms.functional.resize(
            face, size=(height, width), antialias=True
        )
        face = rearrange(face, "c h w -> h w c")
        face = (face / 2 + 0.5).clamp(0, 1)
        face = (face * 255).to(torch.uint8).cpu().numpy()

        out_frame = restorer.restore_img(
            video_frames[index], face, affine_matrices[index]
        )
        out_frames.append(out_frame)

    return np.stack(out_frames, axis=0)


# ======================
# Main functions
# ======================


def recognize_from_video(pipe: df.LipsyncPipeline):
    logger.info("Start inference...")

    synced_video_frames = pipe.forward(
        num_frames=16,
        num_inference_steps=20,
        guidance_scale=1.5,
        height=256,
        width=256,
    )
    synced_video_frames = restore_video(
        synced_video_frames, original_video_frames, boxes, affine_matrices
    )

    video_fps = 25
    audio_sample_rate = 16000
    audio_samples_remain_length = int(
        synced_video_frames.shape[0] / video_fps * audio_sample_rate
    )
    audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()


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
        vae_decoder = ailia.Net(
            MODEL_VAE_DEC_PATH,
            WEIGHT_VAE_DEC_PATH,
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
        vae_decoder = onnxruntime.InferenceSession(
            WEIGHT_VAE_DEC_PATH, providers=providers
        )

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = None
    else:
        raise NotImplementedError

    scheduler = df.schedulers.DDIMScheduler.from_config(
        {
            "beta_end": 0.012,
            "trained_betas": None,
            "beta_start": 0.00085,
            "set_alpha_to_one": False,
            "clip_sample": False,
            "steps_offset": 1,
            "num_train_timesteps": 1000,
            "beta_schedule": "scaled_linear",
        }
    )

    pipe = df.LipsyncPipeline(
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        unet=net,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_video(pipe)


if __name__ == "__main__":
    main()
