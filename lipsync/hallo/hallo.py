import os
import shutil
import subprocess
import sys
import tempfile
from logging import getLogger

import ailia
import cv2
import librosa
import numpy as np
import soundfile as sf
import tqdm

# import original modules
sys.path.append("../../util")
import df
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa
from model_utils import check_and_download_file, check_and_download_models  # noqa

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_REF_UNET_PATH = "reference_unet.onnx"
WEIGHT_DENOISE_PATH = "denoising_unet.onnx"
MODEL_REF_UNET_PATH = "reference_unet.onnx.prototxt"
MODEL_DENOISE_PATH = "denoising_unet.onnx.prototxt"
WEIGHT_DENOISE_PB_PATH = "denoising_unet_weights.pb"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/hallo/"

IMAGE_SIZE = 512
IMAGE_PATH = "demo.jpg"
WAV_PATH = "demo.wav"
VIDEO_PATH = "demo.wav"
SAVE_VIDEO_PATH = "output.mp4"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "Hallo: Hierarchical Audio-Driven Visual Synthesis for Portrait Image Animation",
    IMAGE_PATH,
    SAVE_VIDEO_PATH,
    input_ftype="audio",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)
args.input = parser.parse_args().input


# ======================
# Secondary Functions
# ======================




# ======================
# Main functions
# ======================


class FaceAnimatePipeline:
    def __init__(
        self,
        reference_unet,
        denoising_unet,
        scheduler,
        use_onnx: bool = False,
    ):
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.scheduler = scheduler
        self.use_onnx = use_onnx

    def forward(
        self,
        num_inference_steps,
        guidance_scale,
    ):
        motion_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        do_classifier_free_guidance = guidance_scale > 1.0

        batch_size = 1

        encoder_hidden_states = np.load("encoder_hidden_states.npy")
        uncond_encoder_hidden_states = np.load("uncond_encoder_hidden_states.npy")

        if do_classifier_free_guidance:
            encoder_hidden_states = np.concatenate(
                [uncond_encoder_hidden_states, encoder_hidden_states], axis=0
            )
        latents = np.load("latents.npy")

        ref_image_latents = np.load("ref_image_latents.npy")

        face_mask = (
            np.concatenate([np.zeros_like(face_mask), face_mask], axis=0)
            if do_classifier_free_guidance
            else face_mask
        )

        audio_tensor = np.load("audio_tensor.npy")
        uncond_audio_tensor = np.zeros_like(audio_tensor)
        audio_tensor = np.concatenate([uncond_audio_tensor, audio_tensor], axis=0)

        pixel_values_full_mask = [
            np.load("pixel_values_full_mask_%d.npy" % i) for i in range(4)
        ]
        pixel_values_face_mask = [
            np.load("pixel_values_face_mask_%d.npy" % i) for i in range(4)
        ]
        pixel_values_lip_mask = [
            np.load("pixel_values_lip_mask_%d.npy" % i) for i in range(4)
        ]

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Forward reference image
                if i == 0:
                    # feedforward
                    if not args.onnx:
                        output = self.reference_unet.predict(
                            [
                                np.repeat(
                                    ref_image_latents,
                                    (2 if do_classifier_free_guidance else 1),
                                    axis=0,
                                ),
                                np.zeros_like(t),
                                encoder_hidden_states,
                            ]
                        )
                    else:
                        output = self.reference_unet.run(
                            None,
                            {
                                "sample": np.repeat(
                                    ref_image_latents,
                                    (2 if do_classifier_free_guidance else 1),
                                    axis=0,
                                ),
                                "timestep": np.zeros_like(t),
                                "encoder_hidden_states": encoder_hidden_states,
                            },
                        )
                    _, *bank = output
                bank = [np.load("bank_%d.npy" % i) for i in range(16)]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    np.concatenate([latents] * 2)
                    if do_classifier_free_guidance
                    else latents
                )
                latent_model_input = np.load("latent_model_input.npy")

                if not args.onnx:
                    output = self.denoising_unet.predict(
                        [
                            latent_model_input,
                            np.array(t),
                            encoder_hidden_states,
                            audio_tensor,
                            face_mask,
                            *pixel_values_full_mask,
                            *pixel_values_face_mask,
                            *pixel_values_lip_mask,
                            np.array(motion_scale),
                            *bank,
                        ]
                    )
                else:
                    output = self.denoising_unet.run(
                        None,
                        {
                            "sample": latent_model_input,
                            "timestep": np.array(t),
                            "encoder_hidden_states": encoder_hidden_states,
                            "audio_embedding": audio_tensor,
                            "mask_cond_fea": face_mask,
                            **{
                                "full_mask_%d" % i: x
                                for i, x in enumerate(pixel_values_full_mask)
                            },
                            **{
                                "face_mask_%d" % i: x
                                for i, x in enumerate(pixel_values_face_mask)
                            },
                            **{
                                "lip_mask_%d" % i: x
                                for i, x in enumerate(pixel_values_lip_mask)
                            },
                            "motion_scale": motion_scale,
                            **{"bank_%d" % i: x for i, x in enumerate(bank)},
                        },
                    )
                noise_pred = output[0]

    def progress_bar(self, iterable=None, total=None):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)


def recognize_from_video(pipe: FaceAnimatePipeline):
    logger.info("Start inference...")

    videos = pipe.forward(
        num_inference_steps=40,
        guidance_scale=3.5,
    )

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_REF_UNET_PATH, MODEL_REF_UNET_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DENOISE_PATH, MODEL_DENOISE_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_DENOISE_PB_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        reference_unet = ailia.Net(
            MODEL_REF_UNET_PATH,
            WEIGHT_REF_UNET_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
        denoising_unet = ailia.Net(
            MODEL_DENOISE_PATH,
            WEIGHT_DENOISE_PATH,
            env_id=env_id,
            memory_mode=memory_mode,
        )
    else:
        import onnxruntime

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.use_env_allocators", "1")
        so.intra_op_num_threads = 1

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        reference_unet = onnxruntime.InferenceSession(
            WEIGHT_REF_UNET_PATH, providers=providers
        )
        denoising_unet = onnxruntime.InferenceSession(
            WEIGHT_DENOISE_PATH,
            providers=providers,
            # sess_options=so,
        )

    scheduler = df.schedulers.DDIMScheduler.from_config(
        {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "clip_sample": False,
            "steps_offset": 1,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
        },
    )

    pipe = FaceAnimatePipeline(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_video(pipe)


if __name__ == "__main__":
    main()
