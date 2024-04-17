import sys
import os
import time
import glob
import inspect
from logging import getLogger
from typing import Callable, List, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPTokenizer
from diffusers import UniPCMultistepScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

import ailia
import onnxruntime

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from params import EXTENSIONS

from ootdiffusion_utils.openpose import OpenPose
from ootdiffusion_utils.ootd_parsing import Parsing
from ootdiffusion_utils.ootd_utils import get_mask_location


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_BODY_POSE_PATH = 'body_pose.onnx'
MODEL_BODY_POSE_PATH = 'body_pose.onnx.prototxt'

WEIGHT_ATR_PATH = 'parsing_atr.onnx'
MODEL_ATR_PATH = 'parsing_atr.onnx.prototxt'

WEIGHT_UPSAMPLE_ATR_PATH = 'upsample_atr.onnx'
MODEL_UPSAMPLE_ATR_PATH = 'upsample_atr.onnx.prototxt'

WEIGHT_LIP_PATH = 'parsing_lip.onnx'
MODEL_LIP_PATH = 'parsing_lip.onnx.prototxt'

WEIGHT_UPSAMPLE_LIP_PATH = 'upsample_lip.onnx'
MODEL_UPSAMPLE_LIP_PATH = 'upsample_lip.onnx.prototxt'

WEIGHT_UNET_GARM_HD_PATH = 'unet_garm_hd.onnx'
WEIGHT_UNET_GARM_HD_PB_PATH = 'unet_garm_hd_weights.pb'
MODEL_UNET_GARM_HD_PATH = 'unet_garm_hd.onnx.prototxt'

WEIGHT_UNET_GARM_DC_PATH = 'unet_garm_dc.onnx'
WEIGHT_UNET_GARM_DC_PB_PATH = 'unet_garm_dc_weights.pb'
MODEL_UNET_GARM_DC_PATH = 'unet_garm_dc.onnx.prototxt'

WEIGHT_UNET_VTON_HD_PATH = 'unet_vton_hd.onnx'
WEIGHT_UNET_VTON_HD_PB_PATH = 'unet_vton_hd_weights.pb'
MODEL_UNET_VTON_HD_PATH = 'unet_vton_hd.onnx.prototxt'

WEIGHT_UNET_VTON_DC_PATH = 'unet_vton_hd.onnx'
WEIGHT_UNET_VTON_DC_PB_PATH = 'unet_vton_hd_weights.pb'
MODEL_UNET_VTON_DC_PATH = 'unet_vton_hd.onnx.prototxt'

WEIGHT_INTERPOLATION_PATH = 'ootd_vton_hd_interpolate.onnx'
MODEL_INTERPOLATION_PATH = 'ootd_vton_hd_interpolate.onnx.prototxt'

WEIGHT_TEXT_ENCODER_PATH = 'text_encoder.onnx'
MODEL_TEXT_ENCODER_PATH = 'text_encoder.onnx.prototxt'

WEIGHT_VAE_ENCODER_PATH = 'vae_encoder.onnx'
MODEL_VAE_ENCODER_PATH = 'vae_encoder.onnx.prototxt'

WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'

# WEIGHT_IMAGE_ENCODER_PATH = 'image_encoder.onnx'
# MODEL_IMAGE_ENCODER_PATH = 'image_encoder.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ootdiffusion/'

IMAGE_PATH = 'model.png'
CLOTH_PATH = 'cloth.jpg'
SAVE_IMAGE_PATH = 'output.png'

CATEGORY_DICT = ['upperbody', 'lowerbody', 'dress']
CATEGORY_DICT_UTILS = ['upper_body', 'lower_body', 'dresses']


# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'OOTDiffusion', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-c', '--cloth', nargs='*', metavar='IMAGE/VIDEO', default=CLOTH_PATH,
    help='Path to the cloth image (or folder of images).'
)
parser.add_argument(
    "--model_type", type=str, default="hd", choices=("hd", "dc"),
    help="Model type: 'hd' for half-body model, 'dc' for full-body model."
)
parser.add_argument(
    "--category", type=int, default=0, choices=(0, 1, 2),
    help="Garment category: 0 for upperbody, 1 for lowerbody, 2 for dress."
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Random seed."
)
parser.add_argument(
    "--save_mask", action='store_true',
    help="Save intermediate mask image."
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='Execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

def generate_list_input_cloths(cloth_path):
    if os.path.isdir(cloth_path):
        # Directory Path --> generate list of inputs
        files_grapped = []
        in_dir = cloth_path
        for extension in EXTENSIONS[args.ftype]:
            files_grapped.extend(glob.glob(os.path.join(in_dir, extension)))
        logger.info(f'{len(files_grapped)} {args.ftype} files found!')

        cloth_path = sorted(files_grapped)
    elif os.path.isfile(cloth_path):
        cloth_path = [cloth_path]

    return cloth_path

@dataclass
class OpenPoseModels:
    body_pose: ailia.Net | onnxruntime.InferenceSession | None

    def release_body_pose(self):
        del self.body_pose
        self.body_pose = None

@dataclass
class ParsingModels:
    atr: ailia.Net | onnxruntime.InferenceSession | None
    upsample_atr: ailia.Net | onnxruntime.InferenceSession | None
    lip: ailia.Net | onnxruntime.InferenceSession | None
    upsample_lip: ailia.Net | onnxruntime.InferenceSession | None

    def release_atr(self):
        del self.atr
        self.atr = None

        del self.upsample_atr
        self.upsample_atr = None

    def release_lip(self):
        del self.lip
        self.lip = None

        del self.upsample_lip
        self.upsample_lip = None

@dataclass
class OOTDiffusionModels:
    # ailia/ort models
    text_encoder: ailia.Net | onnxruntime.InferenceSession | None
    vae_encoder: ailia.Net | onnxruntime.InferenceSession | None
    vae_decoder: ailia.Net | onnxruntime.InferenceSession | None
    unet_garm: ailia.Net | onnxruntime.InferenceSession | None
    unet_vton: ailia.Net | onnxruntime.InferenceSession | None
    interpolation: ailia.Net | onnxruntime.InferenceSession | None

    # transformers/diffusers models
    tokenizer: CLIPTokenizer | None
    scheduler: UniPCMultistepScheduler | None
    auto_processor: AutoProcessor | None
    image_encoder: CLIPVisionModelWithProjection | None
    image_processor: VaeImageProcessor | None

    def release_text_encoder(self):
        del self.text_encoder
        self.text_encoder = None

    def release_vae_encoder(self):
        del self.vae_encoder
        self.vae_encoder = None

    def release_vae_decoder(self):
        del self.vae_decoder
        self.vae_decoder = None

    def release_unet_garm(self):
        del self.unet_garm
        self.unet_garm = None

    def release_unet_vton(self):
        del self.unet_vton
        self.unet_vton = None

    def release_interpolation(self):
        del self.interpolation
        self.interpolation = None

    def release_tokenizer(self):
        del self.tokenizer
        self.tokenizer = None

    def release_scheduler(self):
        del self.scheduler
        self.scheduler = None

    def release_auto_processor(self):
        del self.auto_processor
        self.auto_processor = None

    def release_image_encoder(self):
        del self.image_encoder
        self.image_encoder = None

    def release_image_processor(self):
        del self.image_processor
        self.image_processor = None

class OOTDiffusion:
    def __init__(self, models, vae_scale_factor, is_onnx):
        self.models = models

        # self.pipe = OotdPipeline.from_pretrained(
        #     MODEL_PATH,
        #     unet_garm=self.unet_garm,
        #     unet_vton=self.unet_vton,
        #     vae_encoder=self.vae_encoder,
        #     vae_decoder=self.vae_decoder,
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        #     use_safetensors=True,
        #     # safety_checker=None,
        #     # requires_safety_checker=False,
        # ).to(self.gpu_id)

        self.vae_scale_factor = vae_scale_factor
        self.vae_config_scaling_factor = 0.18215
        self.is_onnx = is_onnx

    def tokenize_captions(self, captions, max_length):
        inputs = self.models.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="np"
        )
        return inputs.input_ids.astype(np.int32)

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: Optional[int],
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:  # Skipped
            # get prompt text embeddings
            text_inputs = self.models.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.models.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.models.tokenizer(prompt, padding="longest", return_tensors="np").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.models.tokenizer.batch_decode(
                    untruncated_ids[:, self.models.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    f"The following part of your input was truncated because CLIP can only handle sequences up to \
                    {self.models.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.models.text_encoder.config, "use_attention_mask") and self.models.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None

            prompt_embeds = self.models.text_encoder(input_ids=text_input_ids.astype(np.int32), attention_mask=attention_mask)[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] # * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.models.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            # negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            # negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = np.concatenate([prompt_embeds, prompt_embeds])

        return prompt_embeds

    def _prepare_garm_latents(
        self, image, batch_size, num_images_per_prompt, dtype, do_classifier_free_guidance, generator=None
    ):
        image = image.detach().numpy().astype(dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):  # Skipped
                # image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                if self.is_onnx:
                    image_latents = [self.models.vae_encoder.run(None, {self.models.vae_encoder.get_inputs()[0].name: image[i : i + 1]})[0] for i in range(batch_size)]
                else:
                    image_latents = [self.models.vae_encoder.run(image[i : i + 1])[0] for i in range(batch_size)]
                image_latents = np.concatenate(image_latents, axis=0)
            else:
                # image_latents = self.vae.encode(image).latent_dist.mode()
                if self.is_onnx:
                    image_latents = self.models.vae_encoder.run(None, {self.models.vae_encoder.get_inputs()[0].name: image})[0]
                else:
                    image_latents = self.models.vae_encoder.run(image)[0]

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = np.concatenate([image_latents] * additional_image_per_prompt, axis=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = np.concatenate([image_latents], axis=0)

        if do_classifier_free_guidance:
            uncond_image_latents = np.zeros_like(image_latents)
            image_latents = np.concatenate([image_latents, uncond_image_latents], axis=0)

        return image_latents

    def _prepare_vton_latents(
        self, image, mask, image_ori, batch_size, num_images_per_prompt, dtype, do_classifier_free_guidance, generator=None
    ):
        image = image.detach().numpy().astype(dtype)
        image_ori = image_ori.detach().numpy().astype(dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
            image_ori_latents = image_ori
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                # image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                # image_ori_latents = [self.vae.encode(image_ori[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                if self.is_onnx:
                    image_latents = [self.models.vae_encoder.run(None, {self.models.vae_encoder.get_inputs()[0].name: image[i : i + 1]})[0] for i in range(batch_size)]
                    image_ori_latents = [self.models.vae_encoder.run(None, {self.models.vae_encoder.get_inputs()[0].name: image_ori[i : i + 1]})[0] for i in range(batch_size)]
                else:
                    image_latents = [self.models.vae_encoder.run(image[i : i + 1])[0] for i in range(batch_size)]
                    image_ori_latents = [self.models.vae_encoder.run(image_ori[i : i + 1])[0] for i in range(batch_size)]

                image_latents = np.concatenate(image_latents, axis=0)
                image_ori_latents = np.concatenate(image_ori_latents, axis=0)
            else:
                # image_latents = self.vae.encode(image).latent_dist.mode()
                # image_ori_latents = self.vae.encode(image_ori).latent_dist.mode()
                if self.is_onnx:
                    image_latents = self.models.vae_encoder.run(None, {self.models.vae_encoder.get_inputs()[0].name: image})[0]
                    image_ori_latents = self.models.vae_encoder.run(None, {self.models.vae_encoder.get_inputs()[0].name: image_ori})[0]
                else:
                    image_latents = self.models.vae_encoder.run(image)[0]
                    image_ori_latents = self.models.vae_encoder.run(image_ori)[0]

        if self.is_onnx:
            mask = self.models.interpolation.run(None, {self.models.interpolation.get_inputs()[0].name: mask.astype(np.float32),
                                                        self.models.interpolation.get_inputs()[1].name: np.array(image_latents.shape, dtype=np.int32)})[0]
        else:
            mask = self.models.interpolation.run([mask.astype(np.float32), np.array(image_latents.shape, dtype=np.int32)])[0]

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = np.concatenate([image_latents] * additional_image_per_prompt, axis=0)
            mask = np.concatenate([mask] * additional_image_per_prompt, axis=0)
            image_ori_latents = np.concatenate([image_ori_latents] * additional_image_per_prompt, axis=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = np.concatenate([image_latents], axis=0)
            mask = np.concatenate([mask], axis=0)
            image_ori_latents = np.concatenate([image_ori_latents], axis=0)

        if do_classifier_free_guidance:
            # uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = np.concatenate([image_latents] * 2, axis=0)

        return image_latents, mask, image_ori_latents

    def _prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            # latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = np.random.rand(*shape).astype(dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.models.scheduler.init_noise_sigma
        return latents

    def _prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.models.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.models.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _execute_pipeline(
        self,
        prompt: Union[str, List[str]] = None, # default
        # height: Optional[int] = 512,
        # width: Optional[int] = 512,
        image_garm: PipelineImageInput = None,
        image_vton: PipelineImageInput = None,
        mask: PipelineImageInput = None,
        image_ori: PipelineImageInput = None,
        num_inference_steps: Optional[int] = 100,
        # guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None, # default
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0, # default
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None, # default
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None, # default
        output_type: Optional[str] = "pil", # default
        return_dict: bool = True, # default
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image` or List[`PIL.Image.Image`] or `torch.FloatTensor`):
                `Image`, or tensor representing an image batch which will be upscaled. *
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`np.random.RandomState`, *optional*):
                One or a list of [numpy generator(s)](TODO) to make generation deterministic.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback", None)

        # 0. Check inputs
        print("Step 0")
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds
        )

        if (image_vton is None) or (image_garm is None):
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        print("Step 1")
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        scheduler_is_in_sigma_space = False

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = image_guidance_scale >= 1.0

        # 2. Encode input prompt
        print("Step 2")
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self.models.release_text_encoder()
        self.models.release_tokenizer()

        # get the initial random noise unless the user supplied it
        # latents_dtype = prompt_embeds.dtype
        # latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        # if latents is None:
        #     latents = generator.randn(*latents_shape).astype(latents_dtype)
        # elif latents.shape != latents_shape:
        #     raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # 3. Preprocess image
        print("Step 3")
        image_garm = self.models.image_processor.preprocess(image_garm)
        image_vton = self.models.image_processor.preprocess(image_vton)
        image_ori = self.models.image_processor.preprocess(image_ori)
        mask = np.array(mask)
        mask[mask < 127] = 0
        mask[mask >= 127] = 255
        # mask = torch.tensor(mask)
        mask = mask / 255
        # mask = mask.reshape(-1, 1, mask.size(-2), mask.size(-1))
        mask = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])

        # 4. set timesteps
        print("Step 4")
        self.models.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.models.scheduler.timesteps

        # latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # 5. Prepare Image latents
        print("Step 5")
        garm_latents = self._prepare_garm_latents(
            image_garm,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            do_classifier_free_guidance,
            generator,
        )

        vton_latents, mask_latents, image_ori_latents = self._prepare_vton_latents(
            image_vton,
            mask,
            image_ori,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            do_classifier_free_guidance,
            generator,
        )

        self.models.release_vae_encoder()
        self.models.release_interpolation()

        height, width = vton_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 6. Prepare latent variables
        print("Step 6")
        num_channels_latents = 4
        latents = self._prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        noise = latents.copy()

        # 8. Prepare extra step kwargs.
        print("Step 8")
        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, eta)

        # timestep_dtype = next(
        #     (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        # )
        # timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        # 9. Denoising loop
        print("Step 9")
        num_warmup_steps = len(timesteps) - num_inference_steps * self.models.scheduler.order
        self._num_timesteps = len(timesteps)

        # _, spatial_attn_outputs = self.unet_garm(
        #     garm_latents,
        #     0,
        #     encoder_hidden_states=prompt_embeds,
        #     return_dict=False,
        # )
        if self.is_onnx:
            spatial_attn_outputs = self.models.unet_garm.run(None, {self.models.unet_garm.get_inputs()[0].name: garm_latents,
                                                                    self.models.unet_garm.get_inputs()[1].name: np.array([0], dtype=np.int32),
                                                                    self.models.unet_garm.get_inputs()[2].name: prompt_embeds})[1:]
        else:
            spatial_attn_outputs = self.models.unet_garm.run([garm_latents, np.array([0], dtype=np.int32), prompt_embeds])[1:]

        self.models.release_unet_garm()

        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            # concat latents, image_latents in the channel dimension
            # This code is useless so remove it in order to avoid PyTorch.
            # Cf. https://github.com/huggingface/diffusers/blob/b69fd990ad8026f21893499ab396d969b62bb8cc/src/diffusers/schedulers/scheduling_unipc_multistep.py#L833
            # scaled_latent_model_input = self.models.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t).cpu().numpy()
            latent_vton_model_input = np.concatenate([latent_model_input, vton_latents], axis=1)
            # latent_vton_model_input = scaled_latent_model_input + vton_latents

            spatial_attn_inputs = spatial_attn_outputs.copy()

            # predict the noise residual
            if self.is_onnx:
                onnx_inputs = {self.models.unet_vton.get_inputs()[i+1].name: spatial_attn_inputs[i] for i in range(len(spatial_attn_inputs))}
                onnx_inputs[self.models.unet_vton.get_inputs()[0].name: latent_vton_model_input]
                onnx_inputs[self.models.unet_vton.get_inputs()[17].name: np.array([t], dtype=np.int32)]
                onnx_inputs[self.models.unet_vton.get_inputs()[18].name: prompt_embeds]
                noise_pred = self.models.unet_vton.run(None, onnx_inputs)[0]
            else:
                noise_pred = self.models.unet_vton.run([latent_vton_model_input] + spatial_attn_inputs + [np.array([t], dtype=np.int32), prompt_embeds])[0]

            print(noise_pred.shape)
            exit()

            # Hack:
            # For karras style schedulers the model does classifer free guidance using the
            # predicted_original_sample instead of the noise_pred. So we need to compute the
            # predicted_original_sample here if we are using a karras style scheduler.
            if scheduler_is_in_sigma_space:
                # step_index = (self.models.scheduler.timesteps == t).nonzero()[0].item()
                step_index = np.argwhere(self.models.scheduler.timesteps == t)[0]
                sigma = self.scheduler.sigmas[step_index]
                noise_pred = latent_model_input - sigma * noise_pred

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_text_image, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_text + image_guidance_scale * (noise_pred_text_image - noise_pred_text)

            # Hack:
            # For karras style schedulers the model does classifer free guidance using the
            # predicted_original_sample instead of the noise_pred. But the scheduler.step function
            # expects the noise_pred and computes the predicted_original_sample internally. So we
            # need to overwrite the noise_pred here such that the value of the computed
            # predicted_original_sample is correct.
            if scheduler_is_in_sigma_space:
                noise_pred = (noise_pred - latents) / (-sigma)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.models.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            init_latents_proper = image_ori_latents * self.vae_config_scaling_factor

            # repainting
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = self.models.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - mask_latents) * init_latents_proper + mask_latents * latents

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                vton_latents = callback_outputs.pop("vton_latents", vton_latents)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.models.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.models.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        self.models.release_unet_vton()
        self.models.release_scheduler()

        # latents = 1 / 0.18215 * latents
        # # image = self.vae_decoder(latent_sample=latents)[0]
        # # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        # image = np.concatenate(
        #     [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        # )

        # image = np.clip(image / 2 + 0.5, 0, 1)
        # image = image.transpose((0, 2, 3, 1))

        if not output_type == "latent":
            if self.is_onnx:
                image = self.models.vae_decoder.run(None, {self.models.vae_decoder.get_inputs()[0].name: latents / self.vae_config_scaling_factor})[0]
            else:
                image = self.models.vae_decoder.run(latents / self.vae_config_scaling_factor)[0]
        else:
            image = latents

        self.models.release_vae_decoder()

        do_denormalize = [True] * image.shape[0]
        image = self.models.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.models.release_image_processor()

        return image

    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0
    ):
        #prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
        prompt_image = self.models.auto_processor(images=image_garm, return_tensors="pt")
        self.models.release_auto_processor()

        # print(image_garm.size) # (768, 1024)

        prompt_image = self.models.image_encoder(prompt_image.data['pixel_values']).image_embeds.detach().numpy()
        self.models.release_image_encoder()
        # if self.is_onnx:
        #     prompt_image = self.image_encoder.run(None, {self.image_encoder.get_inputs()[0].name: [],
        #                                                  self.image_encoder.get_inputs()[1].name: prompt_image.data['pixel_values'],
        #                                                  self.image_encoder.get_inputs()[2].name: []})
        # else:
        #     prompt_image = self.image_encoder.run(prompt_image.data['pixel_values'])

        # print(prompt_image.data['pixel_values'].shape)  # torch.Size([1, 3, 224, 224])

        # prompt_image = prompt_image.unsqueeze(1)
        prompt_image = np.expand_dims(prompt_image, axis=1)

        if model_type == 'hd':
            # prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0]
            if self.is_onnx:
                prompt_embeds = self.models.text_encoder.run(None, {self.models.text_encoder.get_inputs()[0].name: self.tokenize_captions([""], 2)})[0]
            else:
                prompt_embeds = self.models.text_encoder.run(self.tokenize_captions([""], 2))[0]
            prompt_embeds[:, 1:] = prompt_image[:]
        elif model_type == 'dc':
            # prompt_embeds = self.text_encoder(self.tokenize_captions([category], 3).to(self.gpu_id))[0]
            if self.is_onnx:
                prompt_embeds = self.models.text_encoder.run(None, {self.models.text_encoder.get_inputs()[0].name: self.tokenize_captions([category], 3)})[0]
            else:
                prompt_embeds = self.models.text_encoder.run(self.tokenize_captions([category], 3))[0]
            # prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            prompt_embeds = np.concatenate([prompt_embeds, prompt_image], axis=1)
        else:
            raise ValueError("model_type must be \'hd\' or \'dc\'!")

        images = self._execute_pipeline(prompt_embeds=prompt_embeds,
                    image_garm=image_garm,
                    image_vton=image_vton,
                    mask=mask,
                    image_ori=image_ori,
                    num_inference_steps=num_steps,
                    image_guidance_scale=image_scale,
                    num_images_per_prompt=num_samples
        )

        return images


# ======================
# Main functions
# ======================

def predict(models, img):
    pass


def recognize_from_image(openpose_pipeline, parsing_pipeline, ootd_pipeline):
    # input image loop
    for image_path, cloth_path in zip(args.input, args.cloth):
        logger.info(image_path)
        image_name = os.path.basename(image_path).split('.')[0]

        # prepare input data
        cloth_img = Image.open(cloth_path).resize((768, 1024))
        model_img = Image.open(image_path).resize((768, 1024))

        # inference
        keypoints = openpose_pipeline(model_img.resize((384, 512)))

        model_parse, _ = parsing_pipeline(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(args.model_type, CATEGORY_DICT_UTILS[args.category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_img, mask)

        if args.save_mask:
            savepath = os.path.join(os.path.dirname(args.savepath), f'{image_name}_mask.jpg')
            logger.info(f'mask image saved at : {savepath}')
            masked_vton_img.save(savepath)

        images = ootd_pipeline(
            model_type=args.model_type,
            category=CATEGORY_DICT[args.category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=1,
            num_steps=20,
            image_scale=2.0,
        )

        savepath = os.path.join(os.path.dirname(args.savepath), f'{image_name}_output.jpg')
        logger.info(f'saved at : {savepath}')
        images[0].save(savepath)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_BODY_POSE_PATH, MODEL_BODY_POSE_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_ATR_PATH, MODEL_ATR_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_UPSAMPLE_ATR_PATH, MODEL_UPSAMPLE_ATR_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_LIP_PATH, MODEL_LIP_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_UPSAMPLE_LIP_PATH, MODEL_UPSAMPLE_LIP_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_UNET_GARM_HD_PATH, MODEL_UNET_GARM_HD_PATH, REMOTE_PATH)
    # check_and_download_file(WEIGHT_UNET_GARM_HD_PB_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_UNET_GARM_DC_PATH, MODEL_UNET_GARM_DC_PATH, REMOTE_PATH)
    # check_and_download_file(WEIGHT_UNET_GARM_DC_PB_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_UNET_VTON_HD_PATH, MODEL_UNET_VTON_HD_PATH, REMOTE_PATH)
    # check_and_download_file(WEIGHT_UNET_VTON_HD_PB_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_UNET_VTON_DC_PATH, MODEL_UNET_VTON_DC_PATH, REMOTE_PATH)
    # check_and_download_file(WEIGHT_UNET_VTON_DC_PB_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_INTERPOLATION_PATH, MODEL_INTERPOLATION_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)

    env_id = args.env_id
    seed = args.seed

    assert len(args.input) == len(args.cloth), \
        f'Arguments --input and --cloth must have the same length, but found {len(args.input)} and {len(args.cloth)} images respectively.'

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(True, True, True, True)
        body_pose = ailia.Net(None, WEIGHT_BODY_POSE_PATH, env_id=env_id, memory_mode=memory_mode)
        atr = ailia.Net(None, WEIGHT_ATR_PATH, env_id=env_id, memory_mode=memory_mode)
        upsample_atr = ailia.Net(None, WEIGHT_UPSAMPLE_ATR_PATH, env_id=env_id, memory_mode=memory_mode)
        lip = ailia.Net(None, WEIGHT_LIP_PATH, env_id=env_id, memory_mode=memory_mode)
        upsample_lip = ailia.Net(None, WEIGHT_UPSAMPLE_LIP_PATH, env_id=env_id, memory_mode=memory_mode)
        interpolation = ailia.Net(None, WEIGHT_INTERPOLATION_PATH, env_id=env_id, memory_mode=memory_mode)
        text_encoder = ailia.Net(None, WEIGHT_TEXT_ENCODER_PATH, env_id=env_id, memory_mode=memory_mode)
        vae_encoder = ailia.Net(None, WEIGHT_VAE_ENCODER_PATH, env_id=env_id, memory_mode=memory_mode)
        vae_decoder = ailia.Net(None, WEIGHT_VAE_DECODER_PATH, env_id=env_id, memory_mode=memory_mode)
        # image_encoder = ailia.Net(None, WEIGHT_IMAGE_ENCODER_PATH, env_id=env_id, memory_mode=None)
        if args.model_type == 'hd':
            unet_garm = ailia.Net(None, WEIGHT_UNET_GARM_HD_PATH, env_id=env_id, memory_mode=memory_mode)
            unet_vton = ailia.Net(None, WEIGHT_UNET_VTON_HD_PATH, env_id=env_id, memory_mode=memory_mode)
        elif args.model_type == 'dc':
            unet_garm = ailia.Net(None, WEIGHT_UNET_GARM_DC_PATH, env_id=env_id, memory_mode=memory_mode)
            unet_vton = ailia.Net(None, WEIGHT_UNET_VTON_DC_PATH, env_id=env_id, memory_mode=memory_mode)
        else:
            print(f'Model "{args.model_type}" is not implemented.')
            exit(-1)
    else:
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        body_pose = onnxruntime.InferenceSession(WEIGHT_BODY_POSE_PATH, providers=providers)
        atr = onnxruntime.InferenceSession(WEIGHT_ATR_PATH, providers=providers)
        upsample_atr = onnxruntime.InferenceSession(WEIGHT_UPSAMPLE_ATR_PATH, providers=providers)
        lip = onnxruntime.InferenceSession(WEIGHT_LIP_PATH, providers=providers)
        upsample_lip = onnxruntime.InferenceSession(WEIGHT_UPSAMPLE_LIP_PATH, providers=providers)
        interpolation = onnxruntime.InferenceSession(WEIGHT_INTERPOLATION_PATH, providers=providers)
        text_encoder = onnxruntime.InferenceSession(WEIGHT_TEXT_ENCODER_PATH, providers=providers)
        vae_encoder = onnxruntime.InferenceSession(WEIGHT_VAE_ENCODER_PATH, providers=providers)
        vae_decoder = onnxruntime.InferenceSession(WEIGHT_VAE_DECODER_PATH, providers=providers)
        # image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_PATH, providers=providers)
        if args.model_type == 'hd':
            unet_garm = onnxruntime.InferenceSession(WEIGHT_UNET_GARM_HD_PATH, providers=providers)
            unet_vton = onnxruntime.InferenceSession(WEIGHT_UNET_VTON_HD_PATH, providers=providers)
        elif args.model_type == 'dc':
            unet_garm = onnxruntime.InferenceSession(WEIGHT_UNET_GARM_DC_PATH, providers=providers)
            unet_vton = onnxruntime.InferenceSession(WEIGHT_UNET_VTON_DC_PATH, providers=providers)
        else:
            print(f'Model "{args.model_type}" is not implemented.')
            exit(-1)

    # tokenizer = CLIPTokenizer.from_pretrained("clip_vit_large_patch14")
    tokenizer = CLIPTokenizer.from_pretrained("processor")

    # scheduler = UniPCMultistepScheduler.from_config("scheduler/scheduler_config.json")
    scheduler = UniPCMultistepScheduler.from_pretrained("scheduler/scheduler_config.json")

    # auto_processor = AutoProcessor.from_pretrained("clip_vit_large_patch14")
    auto_processor = AutoProcessor.from_pretrained("processor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("processor")#.to(self.gpu_id)

    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    openpose_models = OpenPoseModels(body_pose)
    body_pose = None  # Python keeps a reference on this object even though it is not used later. We need to remove it so we can delete the model later.
    openpose_pipeline = OpenPose(openpose_models, args.onnx)

    parsing_models = ParsingModels(atr, upsample_atr, lip, upsample_lip)
    # Python keeps a reference on these objects even though it is not used later.
    # We need to remove it so we can delete the models later.
    atr = None
    upsample_atr = None
    lip = None
    upsample_lip = None
    parsing_pipeline = Parsing(parsing_models, args.onnx)

    ootdiffusion_models = OOTDiffusionModels(text_encoder,
                                             vae_encoder,
                                             vae_decoder,
                                             unet_garm,
                                             unet_vton,
                                             interpolation,
                                             tokenizer,
                                             scheduler,
                                             auto_processor,
                                             image_encoder,
                                             image_processor)
    # Python keeps a reference on these objects even though it is not used later.
    # We need to remove it so we can delete the models later.
    text_encoder = None
    vae_encoder = None
    vae_decoder = None
    unet_garm = None
    unet_vton = None
    interpolation = None
    ootd_pipeline = OOTDiffusion(ootdiffusion_models, vae_scale_factor, args.onnx)

    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)

    recognize_from_image(openpose_pipeline, parsing_pipeline, ootd_pipeline)


if __name__ == '__main__':
    args.cloth = generate_list_input_cloths(args.cloth)
    main()
