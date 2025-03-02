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

from typing import List, Optional, Tuple, Union
from logging import getLogger

import numpy as np
from df.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler

logger = getLogger(__name__)


class StableDiffusion:
    def __init__(
        self,
        vae_decoder,
        text_encoder,
        tokenizer,
        unet,
        scheduler: EulerDiscreteScheduler,
        use_onnx: bool = False,
    ):
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.use_onnx = use_onnx
        self.vae_scale_factor = 8

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, list]],
        do_classifier_free_guidance: bool,
        num_images_per_prompt: int,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`Union[str, List[str]]`):
                prompt to be encoded
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise Exception("Prompt must be either a string or a list of strings.")

        # generate prompt_embeded if not provided
        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="max_length", return_tensors="np", max_length=self.tokenizer.model_max_length
            ).input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            text_input_ids = text_input_ids.astype(np.int32)
            if self.use_onnx:
                prompt_embeds = self.text_encoder.run(
                    None, {"input_ids": text_input_ids}
                )[0]
            else:
                prompt_embeds = self.text_encoder.predict([text_input_ids])[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # generate negative_prompt_embeds if not provided
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
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_input_ids = uncond_input.input_ids.astype(np.int32)
            if self.use_onnx:
                negative_prompt_embeds = self.text_encoder.run(
                    None, {"input_ids": negative_input_ids}
                )[0]
            else:
                negative_prompt_embeds = self.text_encoder.predict(
                    [negative_input_ids]
                )[0]

        # generate prompt_embeds from input
        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(
                negative_prompt_embeds, num_images_per_prompt, axis=0
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: np.dtype,
        latents: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        generator = np.random

        if latents is None:
            latents = generator.randn(*shape).astype(dtype)
        elif latents.shape != shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {shape}"
            )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        return latents

    def _progress_bar(
        self, iterable: Optional[iter] = None, total: Optional[float] = None
    ):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)

    def forward(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, list]] = None,
        num_images_per_prompt: int = 1,
        latents: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        sample_size = 64
        height = height or sample_size * self.vae_scale_factor
        width = width or sample_size * self.vae_scale_factor

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise Exception("Prompt must be either a string or a list of strings.")

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
        )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        in_channels = 4
        latents = self._prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=in_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            latents=latents,
        )
        for i, t in enumerate(self._progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timestep = np.array([t], dtype=int)
            if self.use_onnx:
                noise_pred = self.unet.run(
                    None,
                    {
                        "sample": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": prompt_embeds,
                    },
                )[0]
            else:
                noise_pred = self.unet.predict(
                    [latent_model_input, timestep, prompt_embeds],
                )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )

        latents /= 0.18215
        outputs = []
        for i in range(latents.shape[0]):
            latent_sample = latents[i : i + 1]
            if self.use_onnx:
                output = self.vae_decoder.run(None, {"latent_sample": latent_sample})[0]
            else:
                output = self.vae_decoder.predict([latent_sample])[0]
            outputs.append(output)
        image = np.concatenate(outputs)
        # post process
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        return image
