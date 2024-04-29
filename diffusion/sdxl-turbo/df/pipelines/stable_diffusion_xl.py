# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, List, Optional, Tuple, Union
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


class StableDiffusionXL:
    def __init__(
        self,
        vae_decoder,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        use_onnx: bool = False,
    ):
        super().__init__()

        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.scheduler = scheduler
        self.use_onnx = use_onnx

        self.vae_scale_factor = 8

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, list]],
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`Union[str, List[str]]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        if prompt_embeds is None:
            prompt_embeds_list = []
            for tokenizer, text_encoder, input_dtype in zip(
                tokenizers, text_encoders, [np.int32, np.int64]
            ):
                # get prompt text embeddings
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="np",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(
                    prompt, padding="longest", return_tensors="np"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not np.array_equal(text_input_ids, untruncated_ids):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                text_input_ids = text_input_ids.astype(input_dtype)
                if not self.use_onnx:
                    prompt_embeds = text_encoder.predict([text_input_ids])
                else:
                    prompt_embeds = text_encoder.run(
                        None, {"input_ids": text_input_ids}
                    )

                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds[-2]
                prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = np.concatenate(prompt_embeds_list, axis=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = np.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = np.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="np",
                )
                negative_prompt_embeds = text_encoder(
                    input_ids=uncond_input.input_ids.astype(
                        text_encoder.input_dtype.get("input_ids", np.int32)
                    )
                )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds[-2]

                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                negative_prompt_embeds = np.repeat(
                    negative_prompt_embeds, num_images_per_prompt, axis=0
                )
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                negative_prompt_embeds_list.append(negative_prompt_embeds)
            negative_prompt_embeds = np.concatenate(
                negative_prompt_embeds_list, axis=-1
            )

        pooled_prompt_embeds = np.repeat(
            pooled_prompt_embeds, num_images_per_prompt, axis=0
        )
        negative_pooled_prompt_embeds = np.repeat(
            negative_pooled_prompt_embeds, num_images_per_prompt, axis=0
        )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = np.random.randn(*shape).astype(dtype)
        elif latents.shape != shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {shape}"
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        return latents

    def forward(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        # Default height and width to unet
        sample_size = 64
        height = height or sample_size * self.vae_scale_factor
        width = width or sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        in_channels = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            in_channels,
            height,
            width,
            prompt_embeds.dtype,
            latents,
        )

        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = (original_size + crops_coords_top_left + target_size,)
        add_time_ids = np.array(add_time_ids, dtype=prompt_embeds.dtype)

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate(
                (negative_prompt_embeds, prompt_embeds), axis=0
            )
            add_text_embeds = np.concatenate(
                (negative_pooled_prompt_embeds, add_text_embeds), axis=0
            )
            add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
        add_time_ids = np.repeat(
            add_time_ids, batch_size * num_images_per_prompt, axis=0
        )

        # Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            timestep = np.array([t], dtype=int)
            if not self.use_onnx:
                output = self.unet.predict(
                    [latent_model_input, timestep, prompt_embeds, add_text_embeds]
                )
            else:
                output = self.unet.run(
                    None,
                    {
                        "sample": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": prompt_embeds,
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    },
                )
            noise_pred = output[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)

        latents /= 0.13025
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        outputs = []
        for i in range(latents.shape[0]):
            latent_sample = latents[i : i + 1]
            if not self.use_onnx:
                output = self.vae_decoder.predict([latent_sample])
            else:
                output = self.vae_decoder.run(None, {"latent_sample": latent_sample})
            outputs.append(output[0])
        image = np.concatenate(outputs)

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = image[:, :, :, ::-1]  # RGB->BGR

        return image

    def progress_bar(self, iterable=None, total=None):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)
