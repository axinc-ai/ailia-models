from typing import List, Optional, Union

import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config


class DDIMScheduler(ConfigMixin):
    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            clip_sample: bool = True,
            set_alpha_to_one: bool = True,
            steps_offset: int = 0,
            prediction_type: str = "epsilon",
            thresholding: bool = False,
            dynamic_thresholding_ratio: float = 0.995,
            clip_sample_range: float = 1.0,
            sample_max_value: float = 1.0,
            timestep_spacing: str = "leading",
            rescale_betas_zero_snr: bool = False,
    ):
        # if trained_betas is not None:
        #     self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # elif beta_schedule == "linear":
        #     self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # elif beta_schedule == "scaled_linear":
        #     # this schedule is very specific to the latent diffusion model.
        #     self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
        #                                 dtype=torch.float32) ** 2
        # elif beta_schedule == "squaredcos_cap_v2":
        #     # Glide cosine schedule
        #     self.betas = betas_for_alpha_bar(num_train_timesteps)
        # else:
        #     raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
        #
        # # Rescale for zero SNR
        # if rescale_betas_zero_snr:
        #     self.betas = rescale_zero_terminal_snr(self.betas)
        #
        # self.alphas = 1.0 - self.betas
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        #
        # # At every step in ddim, we are looking into the previous alphas_cumprod
        # # For the final step, there is no previous alphas_cumprod because we are already at 0
        # # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # # whether we use the final alpha of the "non-previous" one.
        # self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        #
        # # standard deviation of the initial noise distribution
        # self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1].copy().astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = timesteps

    # def step(
    #         self,
    #         model_output: torch.FloatTensor,
    #         timestep: int,
    #         sample: torch.FloatTensor,
    #         eta: float = 0.0,
    #         use_clipped_model_output: bool = False,
    #         generator=None,
    #         variance_noise: Optional[torch.FloatTensor] = None,
    #         return_dict: bool = True,
    # ) -> Union[DDIMSchedulerOutput, Tuple]:
    #     """
    #     Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    #     process from the learned model outputs (most often the predicted noise).
    #     """
    #     if self.num_inference_steps is None:
    #         raise ValueError(
    #             "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
    #         )
    #
    #     # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    #     # Ideally, read DDIM paper in-detail understanding
    #
    #     # Notation (<variable name> -> <name in paper>
    #     # - pred_noise_t -> e_theta(x_t, t)
    #     # - pred_original_sample -> f_theta(x_t, t) or x_0
    #     # - std_dev_t -> sigma_t
    #     # - eta -> η
    #     # - pred_sample_direction -> "direction pointing to x_t"
    #     # - pred_prev_sample -> "x_t-1"
    #
    #     # 1. get previous step value (=t-1)
    #     prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    #
    #     # 2. compute alphas, betas
    #     alpha_prod_t = self.alphas_cumprod[timestep]
    #     alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    #
    #     beta_prod_t = 1 - alpha_prod_t
    #
    #     # 3. compute predicted original sample from predicted noise also called
    #     # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    #     if self.config.prediction_type == "epsilon":
    #         pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    #         pred_epsilon = model_output
    #     elif self.config.prediction_type == "sample":
    #         pred_original_sample = model_output
    #         pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    #     elif self.config.prediction_type == "v_prediction":
    #         pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
    #         pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    #     else:
    #         raise ValueError(
    #             f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
    #             " `v_prediction`"
    #         )
    #
    #     # 4. Clip or threshold "predicted x_0"
    #     if self.config.thresholding:
    #         pred_original_sample = self._threshold_sample(pred_original_sample)
    #     elif self.config.clip_sample:
    #         pred_original_sample = pred_original_sample.clamp(
    #             -self.config.clip_sample_range, self.config.clip_sample_range
    #         )
    #
    #     # 5. compute variance: "sigma_t(η)" -> see formula (16)
    #     # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    #     variance = self._get_variance(timestep, prev_timestep)
    #     std_dev_t = eta * variance ** (0.5)
    #
    #     if use_clipped_model_output:
    #         # the pred_epsilon is always re-derived from the clipped x_0 in Glide
    #         pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    #
    #     # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    #     pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon
    #
    #     # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    #     prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    #
    #     if eta > 0:
    #         if variance_noise is not None and generator is not None:
    #             raise ValueError(
    #                 "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
    #                 " `variance_noise` stays `None`."
    #             )
    #
    #         if variance_noise is None:
    #             variance_noise = randn_tensor(
    #                 model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
    #             )
    #         variance = std_dev_t * variance_noise
    #
    #         prev_sample = prev_sample + variance
    #
    #     if not return_dict:
    #         return (prev_sample,)
    #
    #     return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
