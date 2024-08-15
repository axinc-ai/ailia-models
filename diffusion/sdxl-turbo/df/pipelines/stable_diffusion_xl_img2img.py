#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List, Optional, Tuple, Union
from logging import getLogger

import numpy as np
from PIL import Image

from .stable_diffusion_xl import StableDiffusionXL

logger = getLogger(__name__)


class StableDiffusionXLImg2Img(StableDiffusionXL):
    def __init__(
        self,
        vae_encoder,
        vae_decoder,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        use_onnx: bool = False,
    ):
        super().__init__(
            vae_decoder,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            scheduler,
            use_onnx,
        )
        self.vae_encoder = vae_encoder

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
    ):
        add_time_ids = (original_size + crops_coords_top_left + target_size,)
        add_neg_time_ids = (original_size + crops_coords_top_left + target_size,)

        add_time_ids = np.array(add_time_ids, dtype=dtype)
        add_neg_time_ids = np.array(add_neg_time_ids, dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt):
        batch_size = batch_size * num_images_per_prompt

        if not self.use_onnx:
            output = self.vae_encoder.run([image])
        else:
            output = self.vae_encoder.run(
                None,
                {"sample": image},
            )
        init_latents = output[0]

        scaling_factor = 0.13025
        init_latents *= scaling_factor
        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = np.concatenate(
                [init_latents] * additional_image_per_prompt, axis=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = np.concatenate([init_latents], axis=0)

        # add noise to latents using the timesteps
        noise = np.random.randn(*init_latents.shape)
        init_latents = self.scheduler.add_noise(
            init_latents,
            noise,
            timestep,
        )
        return init_latents

    def image_processor(self, image):
        height, width = image.shape[:2]
        width, height = (x - x % self.vae_scale_factor for x in (width, height))

        # resize
        image = np.array(
            Image.fromarray(image).resize((width, height), Image.Resampling.BICUBIC)
        )

        image = image / 255
        image = image.astype(np.float32)

        # reshape
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = np.expand_dims(image, axis=0)
        # normalize
        image = 2.0 * image - 1.0

        return image

    def forward(
        self,
        prompt: Union[str, List[str]],
        image: np.ndarray = None,
        strength: float = 0.3,
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

        # Preprocess image
        image = self.image_processor(image)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength
        )
        latent_timestep = np.repeat(
            timesteps[:1], batch_size * num_images_per_prompt, axis=0
        )

        # Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
        )
        latents = latents.astype(prompt_embeds.dtype)

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
        )

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

        # latents = np.load("latents.npy").astype(np.float32)
        # prompt_embeds = np.load("prompt_embeds.npy")
        # add_text_embeds = np.load("add_text_embeds.npy")
        # add_time_ids = np.load("add_time_ids.npy")

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
                    [latent_model_input, timestep, prompt_embeds, add_text_embeds, add_time_ids]
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

        # latents = np.load("latents_out.npy").astype(np.float32)
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

        return image
