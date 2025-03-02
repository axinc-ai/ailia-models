# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

from typing import List, Optional, Tuple, Union
from logging import getLogger

import numpy as np
import scipy
import tqdm

from image_processor import preprocess_fixed_mask_image

logger = getLogger(__name__)


class LipsyncPipeline:
    def __init__(
        self,
        vae_encoder,
        vae_decoder,
        tokenizer,
        unet,
        scheduler,
        use_onnx: bool = False,
    ):
        super().__init__()

        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.use_onnx = use_onnx

        self.vae_scale_factor = 8
        self.scaling_factor = 0.18215
        self.shift_factor = 0

    def decode_latents(self, latents):
        latents = latents / self.scaling_factor + self.shift_factor

        # b c f h w -> (b f) c h w
        b, c, f, h, w = latents.shape
        latents = latents.transpose(0, 2, 1, 3, 4)
        latents = latents.reshape(b * f, c, h, w)

        if not self.use_onnx:
            output = self.vae_decoder.predict([latents])
        else:
            output = self.vae_decoder.run(None, {"latents": latents})
        decoded_latents = output[0]

        return decoded_latents

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

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        height,
        width,
        do_classifier_free_guidance,
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        batch, channel, height, width = mask.shape
        new_height = height // self.vae_scale_factor
        new_width = width // self.vae_scale_factor
        resized = np.zeros((batch, channel, new_height, new_width), dtype=mask.dtype)
        for b in range(batch):
            for c in range(channel):
                resized[b, c] = scipy.ndimage.zoom(
                    mask[b, c],
                    (new_height / height, new_width / width),
                    order=1,
                    mode="grid-constant",
                    grid_mode=True,
                )
        mask = resized

        # encode the mask image into latents space so we can concatenate it to the latents
        if not self.use_onnx:
            output = self.vae_encoder.predict([masked_image])
        else:
            output = self.vae_encoder.run(None, {"x": masked_image})
        moments = output[0]

        # sample from the latent distribution
        mean, logvar = np.split(moments, 2, axis=1)
        logvar = np.clip(logvar, -30.0, 20.0)
        std = np.exp(0.5 * logvar)
        sample = np.random.randn(*mean.shape)
        masked_image_latents = mean + std * sample

        masked_image_latents = (
            masked_image_latents - self.shift_factor
        ) * self.scaling_factor

        # assume batch size = 1
        # f c h w -> 1 c f h w
        mask = mask.transpose(1, 0, 2, 3)[None]
        masked_image_latents = masked_image_latents.transpose(1, 0, 2, 3)[None]

        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            np.concatenate([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, do_classifier_free_guidance):
        if not self.use_onnx:
            output = self.vae_encoder.predict([images])
        else:
            output = self.vae_encoder.run(None, {"x": images})
        moments = output[0]

        # sample from the latent distribution
        mean, logvar = np.split(moments, 2, axis=1)
        logvar = np.clip(logvar, -30.0, 20.0)
        std = np.exp(0.5 * logvar)
        sample = np.random.randn(*mean.shape)
        image_latents = mean + std * sample

        image_latents = (image_latents - self.shift_factor) * self.scaling_factor

        # f c h w -> 1 c f h w
        image_latents = image_latents.transpose(1, 0, 2, 3)[None]
        image_latents = (
            np.concatenate([image_latents] * 2)
            if do_classifier_free_guidance
            else image_latents
        )

        return image_latents

    def forward(
        self,
        num_frames: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        # negative_prompt: Optional[Union[str, List[str]]] = None,
        # num_images_per_prompt: int = 1,
        # latents: Optional[np.ndarray] = None,
        # prompt_embeds: Optional[np.ndarray] = None,
        # negative_prompt_embeds: Optional[np.ndarray] = None,
        # pooled_prompt_embeds: Optional[np.ndarray] = None,
        # negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        # original_size: Optional[Tuple[int, int]] = None,
        # crops_coords_top_left: Tuple[int, int] = (0, 0),
        # target_size: Optional[Tuple[int, int]] = None,
    ):
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        video_fps = 25
        num_inferences = 15

        # Prepare latent variables
        # all_latents = self.prepare_latents(
        #     batch_size,
        #     num_frames * num_inferences,
        #     num_channels_latents,
        #     height,
        #     width,
        #     weight_dtype,
        #     device,
        #     generator,
        # )

        # Denoising loop
        synced_video_frames = []
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            audio_embeds = np.stack(
                whisper_chunks[i * num_frames : (i + 1) * num_frames]
            )
            # audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
            if do_classifier_free_guidance:
                null_audio_embeds = np.zeros_like(audio_embeds)
                audio_embeds = np.concatenate([null_audio_embeds, audio_embeds])
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]

            # prepare_masks_and_masked_images
            results = [preprocess_fixed_mask_image(image) for image in inference_faces]
            pixel_values_list, masked_pixel_values_list, masks_list = list(
                zip(*results)
            )
            pixel_values, masked_pixel_values, masks = (
                np.stack(pixel_values_list),
                np.stack(masked_pixel_values_list),
                np.stack(masks_list),
            )
            pixel_values = pixel_values.astype(np.float16)
            masked_pixel_values = masked_pixel_values.astype(np.float16)

            # Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                do_classifier_free_guidance,
            )

            # Prepare image latents
            image_latents = self.prepare_image_latents(
                pixel_values,
                do_classifier_free_guidance,
            )

            # Denoising loop
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * self.scheduler.order
            )
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        np.concatenate([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    latent_model_input = np.concatenate(
                        [
                            latent_model_input,
                            mask_latents,
                            masked_image_latents,
                            image_latents,
                        ],
                        axis=1,
                    ).astype(np.float16)

                    # predict the noise residual
                    timestep = np.array(t, dtype=int)
                    if not self.use_onnx:
                        output = self.unet.run(
                            [
                                latent_model_input,
                                timestep,
                                audio_embeds,
                            ]
                        )
                    else:
                        output = self.unet.run(
                            None,
                            {
                                "sample": latent_model_input,
                                "timestep": timestep,
                                "encoder_hidden_states": audio_embeds,
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

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or (
                        (j + 1) > num_warmup_steps
                        and (j + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = decoded_latents * (1 - masks) + pixel_values * masks
            synced_video_frames.append(decoded_latents)

        return np.concatenate(synced_video_frames)

    def progress_bar(self, iterable=None, total=None):
        from tqdm.auto import tqdm

        if iterable is not None:
            return tqdm(iterable)
        elif total is not None:
            return tqdm(total=total)
