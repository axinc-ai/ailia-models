from tqdm.auto import tqdm
import cv2

import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Any, Dict


class StableDiffusionXLControlNetPipeline:
    def __init__(
        self,
        vae_decoder,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        controlnet,
        image_proj_model,
        scheduler,
        use_onnx,
    ):
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.controlnet = controlnet
        self.image_proj_model = image_proj_model
        self.scheduler = scheduler
        self.use_onnx = use_onnx
        self.vae_scale_factor = 8

    def _encode_prompt(
        self,
        prompt: str,
        prompt_2: str | None = None,
        prompt_embeds: np.ndarray | None = None,
        negative_prompt_embeds: np.ndarray | None = None,
        pooled_prompt_embeds: np.ndarray | None = None,
        negative_pooled_prompt_embeds: np.ndarray | None = None,
    ):
        prompt = [prompt]
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]

            for i, (prompt, tokenizer, text_encoder) in enumerate(zip(prompts, tokenizers, text_encoders)):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="np"
                )
                text_input_ids = text_inputs.input_ids

                if self.use_onnx:
                    prompt_embeds = text_encoder.run(
                        None,
                        {text_encoder.get_inputs()[0].name: text_input_ids},
                    )
                else:
                    prompt_embeds = text_encoder.predict(
                        [text_input_ids]
                    )

                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = np.concatenate(prompt_embeds_list, axis=2)
            bs_embed, seq_len, _ = prompt_embeds.shape

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        )

    def prepare_latests(self, batch_size, num_channels_latests, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latests, height // self.vae_scale_factor, width //self.vae_scale_factor)

        if latents is None:
            latents = np.random.randn(*shape).astype(dtype)
    
        latents = latents * np.float32(self.scheduler.init_noise_sigma)
        return latents

    def forward(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: np.ndarray = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        image_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        control_guidance_start = [0.0]
        control_guidance_end = [1.0]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self._encode_prompt(
            prompt,
            prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        prompt_image_emb = self._encode_prompt_image_emb(image_embeds)
        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.reshape(bs_embed, seq_len, -1)
        image = np.array(image).astype(np.uint8)
        image = self._prepare_image(
            image=image,
            height=height,
            width=width,
        )
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        latents = self._prepare_latents()

        controlnet_keep = []
        for i in range(self._num_timesteps):
            keeps = [
                1.0 - float(i / self._num_timesteps < s or (i + 1) / self._num_timesteps > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0]
            )

        target_size = (height, width)
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = 1280

        add_time_ids = self._get_add_time_ids((1024, 960), (0, 0), (1024, 960))
        encoder_hidden_states = np.concatenate(
            [prompt_embeds, prompt_image_emb], axis=1
        )
        num_warmup_steps = self._num_timesteps - num_inference_steps * self.scheduler.order

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }

                controlnet_added_cond_kwargs = added_cond_kwargs
                controlnet_cond_scale = controlnet_conditioning_scale
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
                down_block_res_samples, mid_block_res_sample = self._predict_controlnet(
                    t,
                    latent_model_input,
                    prompt_image_emb,
                    image,
                    cond_scale,
                    controlnet_added_cond_kwargs,
                )
                noise_pred = self._predict_unet(
                    t,
                    latent_model_input,
                    encoder_hidden_states,
                    down_block_res_samples,
                    mid_block_res_sample,
                    added_cond_kwargs
                )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )

                if i == self._num_timesteps -1 or (
                    (i+1) > num_warmup_steps and (i+1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        image = self._predict_vae_decoder(latents / 0.13025)
        image = self._postprocess(image)
        return image[0]

    def _predict_vae_decoder(self, v: np.ndarray):
        if self.use_onnx:
            result = self.vae_decoder.run(
                None,
                {
                    self.vae_decoder.get_inputs()[0].name: v.astype(np.float32)
                }
            )[0]
        else:
            result = self.vae_decoder.predict([v])

        return result

    def _predict_controlnet(self, t, control_model_input, prompt_image_emb, image, cond_scale, controlnet_added_cond_kwargs):
        if self.use_onnx:
            sess_input_names = [i.name for i in self.controlnet.get_inputs()]
            inputs = {
                sess_input_names[0]: control_model_input.astype(np.float32),
                sess_input_names[1]: np.array([t]).astype(np.int64),
                sess_input_names[2]: prompt_image_emb.astype(np.float32),
                sess_input_names[3]: image.reshape(1, *image.shape).transpose(0, 3, 1, 2).astype(np.float32),
                sess_input_names[4]: np.array([cond_scale]).astype(np.double),
                sess_input_names[5]: controlnet_added_cond_kwargs["text_embeds"].astype(np.float32),
                sess_input_names[6]: controlnet_added_cond_kwargs["time_ids"].astype(np.float32),
            }
            result = self.controlnet.run(None, inputs)
            # down_block_res_samples, mid_block_res_sample = result[:-1], result[-1]
        else:
            result = self.controlnet.predict([
                control_model_input.astype(np.float32),
                np.array([t]).astype(np.int64),
                prompt_image_emb.astype(np.float32),
                image.reshape(1, *image.shape).transpose(0, 3, 1, 2).astype(np.float32),
                np.array([cond_scale]).astype(np.double),
                controlnet_added_cond_kwargs["text_embeds"].astype(np.float32),
                controlnet_added_cond_kwargs["time_ids"].astype(np.float32),
            ])


        down_block_res_samples, mid_block_res_sample = result[:-1], result[-1]
        return down_block_res_samples, mid_block_res_sample

    def _predict_unet(
            self,
            t,
            latent_model_input,
            encoder_hidden_states, down_block_additional_residuals, mid_block_additional_residual,
            added_cond_kwargs):
        if self.use_onnx:
            sess_input_names = [i.name for i in self.unet.get_inputs()]
            inputs = {
                sess_input_names[0]: latent_model_input.astype(np.float32),
                sess_input_names[1]: np.array([t]).astype(np.int64),
                sess_input_names[2]: encoder_hidden_states.astype(np.float32),
                sess_input_names[3]: down_block_additional_residuals[0].astype(np.float32),
                sess_input_names[4]: down_block_additional_residuals[1].astype(np.float32),
                sess_input_names[5]: down_block_additional_residuals[2].astype(np.float32),
                sess_input_names[6]: down_block_additional_residuals[3].astype(np.float32),
                sess_input_names[7]: down_block_additional_residuals[4].astype(np.float32),
                sess_input_names[8]: down_block_additional_residuals[5].astype(np.float32),
                sess_input_names[9]: down_block_additional_residuals[6].astype(np.float32),
                sess_input_names[10]: down_block_additional_residuals[7].astype(np.float32),
                sess_input_names[11]: down_block_additional_residuals[8].astype(np.float32),
                sess_input_names[12]: mid_block_additional_residual.astype(np.float32),
                sess_input_names[13]: added_cond_kwargs["time_ids"].astype(np.float32),
                sess_input_names[14]: added_cond_kwargs["text_embeds"].astype(np.float32),
            }

            for item in inputs.values():
                print(f"{item.shape=}")

            input(">>>")
            result = self.unet.run(None, inputs)
        else:
            result = self.unet.predict([
                latent_model_input.astype(np.float32),
                np.array([t]).astype(np.int64),
                encoder_hidden_states.astype(np.float32),
                added_cond_kwargs["text_embeds"].astype(np.float32),
                added_cond_kwargs["time_ids"].astype(np.float32),
                down_block_additional_residuals[0].astype(np.float32),
                down_block_additional_residuals[1].astype(np.float32),
                down_block_additional_residuals[2].astype(np.float32),
                down_block_additional_residuals[3].astype(np.float32),
                down_block_additional_residuals[4].astype(np.float32),
                down_block_additional_residuals[5].astype(np.float32),
                down_block_additional_residuals[6].astype(np.float32),
                down_block_additional_residuals[7].astype(np.float32),
                down_block_additional_residuals[8].astype(np.float32),
                mid_block_additional_residual.astype(np.float32)
            ])

    def _encode_prompt_image_emb(self, prompt_image_emb):
        prompt_image_emb = prompt_image_emb.reshape(
            [1, 1, 512]
        )
        if self.use_onnx:
            prompt_image_emb = self.image_proj_model.run(
                None,
                {self.image_proj_model.get_inputs()[0].name: prompt_image_emb},
            )[0]
        else:
            prompt_image_emb = self.image_proj_model.predict([prompt_image_emb])[0]

        return prompt_image_emb

    def _prepare_image(self, image: np.ndarray, height, width):
        height, width = self._get_default_height_width(image, height, width)
        image = self._resize(image, height, width)
        return image

    def _get_default_height_width(self, image, height, width):
        if height is None:
            height = image.shape[0]
        if width is None:
            width = image.shape[1]

        width = width - width % self.vae_scale_factor
        height = height - height % self.vae_scale_factor
        return height, width

    def _resize(self, image, height, width):
        return cv2.resize(image, (width, height))

    def _prepare_latents(self):
        return np.random.randn(1, 4, 128, 120)

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return np.array([add_time_ids])

    def _progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)

    def _postprocess(self, image):
        image = self._denormalize(image)
        return image

    def _denormalize(self, image):
        return (image / 2 + 0.5).clip(0, 1)