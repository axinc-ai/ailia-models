import numpy as np
from configuration_utils import ConfigMixin, register_to_config
from typing import Optional, List, Union


class LCMScheduler(ConfigMixin):
    """
    `LCMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. [`~ConfigMixin`] takes care of storing all config
    attributes that are passed in the scheduler's `__init__` function, such as `num_train_timesteps`. They can be
    accessed via `scheduler.config.num_train_timesteps`. [`SchedulerMixin`] provides general loading and saving
    functionality via the [`SchedulerMixin.save_pretrained`] and [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: np.ndarray | list[float] | None = None,
        set_alpha_to_one: bool = True,
    ):
        if trained_betas is not None:
            self.betas = np.tensor(trained_betas, dtype=np.float32)
        elif beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = np.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        self.custom_timesteps = False

        self._step_index = None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, np.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    @property
    def step_index(self):
        return self._step_index

    def scale_model_input(self, sample: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`np.ndarray`):
                The input sample.
        Returns:
            `np.ndarray`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        original_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        strength: int = 1.0,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps on the training/distillation timestep
                schedule is used. If `timesteps` is passed, `num_inference_steps` must be `None`.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        # 1. Calculate the LCM original training/distillation timestep schedule.
        original_steps = 50

        if original_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        # LCM Timesteps Setting
        # The skipping step parameter k from the paper.
        k = self.config.num_train_timesteps // original_steps
        # LCM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1

        # 2. Calculate the LCM inference timestep schedule.
        if timesteps is not None:
            # 2.1 Handle custom timestep schedules.
            train_timesteps = set(lcm_origin_timesteps)
            non_train_timesteps = []
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

                if timesteps[i] not in train_timesteps:
                    non_train_timesteps.append(timesteps[i])

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.num_inference_steps = len(timesteps)
            self.custom_timesteps = True

            # Apply strength (e.g. for img2img pipelines) (see StableDiffusionImg2ImgPipeline.get_timesteps)
            init_timestep = min(int(self.num_inference_steps * strength), self.num_inference_steps)
            t_start = max(self.num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start * self.order :]
            # TODO: also reset self.num_inference_steps?
        else:
            # 2.2 Create the "standard" LCM inference timestep schedule.
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            skipping_step = len(lcm_origin_timesteps) // num_inference_steps

            if skipping_step < 1:
                raise ValueError(
                    f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}."
                )

            self.num_inference_steps = num_inference_steps

            if num_inference_steps > original_steps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                    f" {original_steps} because the final timestep schedule will be a subset of the"
                    f" `original_inference_steps`-sized initial timestep schedule."
                )

            # LCM Inference Steps Schedule
            lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
            # Select (approximately) evenly spaced indices from lcm_origin_timesteps.
            inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            timesteps = lcm_origin_timesteps[inference_indices]

        self.timesteps = timesteps
        self._step_index = None

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        self.sigma_data = 0.5  # Default: 0.5
        scaled_timestep = timestep * self.config.timestep_scaling

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
    ) -> tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`np.ndarray`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        Returns:
            `tuple`:
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        if self.config.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )

        if self.config.thresholding:
            predicted_original_sample = self._threshold_sample(predicted_original_sample)
        elif self.config.clip_sample:
            predicted_original_sample = predicted_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        denoised = c_out * predicted_original_sample + c_skip * sample
        prev_sample = denoised
        self._step_index += 1
        return (prev_sample, denoised)