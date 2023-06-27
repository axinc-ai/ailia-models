# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import Callable, List, Optional, Union
from logging import getLogger

import numpy as np
import torch

# from ...schedulers import PNDMScheduler


logger = getLogger(__name__)


class StableDiffusion:
    def __init__(
            self,
            # vae_encoder: OnnxRuntimeModel,
            vae_decoder,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            use_onnx: bool = False,
    ):
        super().__init__()

        # self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.use_onnx = use_onnx

    def _encode_prompt(
            self,
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

        if not np.array_equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if not self.use_onnx:
            output = self.text_encoder.predict([text_input_ids])
        else:
            output = self.text_encoder.run(None, {'input_ids': text_input_ids.astype(np.int32)})
        prompt_embeds = output[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
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
            if not self.use_onnx:
                output = self.text_encoder.predict([uncond_input.input_ids])
            else:
                output = self.text_encoder.run(None, {'input_ids': uncond_input.input_ids.astype(np.int32)})
            negative_prompt_embeds = output[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def forward(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
    ):
        # define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        # elif isinstance(prompt, list):
        else:
            batch_size = len(prompt)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        prompt_embeds = np.load("prompt_embeds.npy").astype(np.float32)

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        latents = np.random.randn(*latents_shape).astype(latents_dtype)
        latents = np.load("latents.npy").astype(latents_dtype)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Denoising loop
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timestep = np.array([t], dtype=np.float32)

            # predict the noise residual
            if not self.use_onnx:
                output = self.unet.predict([latent_model_input, timestep, prompt_embeds])
            else:
                output = self.unet.run(None, {
                    'sample': latent_model_input,
                    'timestep': timestep,
                    'encoder_hidden_states': prompt_embeds})
            noise_pred = output[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            )

        latents = latents / 0.18215
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        outputs = []
        for i in range(latents.shape[0]):
            latent_sample = latents[i: i + 1]
            if not self.use_onnx:
                output = self.vae_decoder.predict([latent_model_input])
            else:
                output = self.vae_decoder.run(None, {'latent_sample': latent_sample})
            outputs.append(output[0])
        image = np.concatenate(outputs)

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        return image

    def progress_bar(self, iterable=None, total=None):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)
