# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union

import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config


class EulerDiscreteScheduler(ConfigMixin):

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
    ):
        if trained_betas is not None:
            self.betas = np.array(trained_betas, dtype=np.float32)
        elif beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.flip(sigmas, axis=0)
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()

        self.sigmas = np.concatenate((sigmas, np.zeros(1)), axis=0)
        self.timesteps = timesteps
        self.num_inference_steps = None        
        self.is_scale_input_called = False
        self._step_index = None
        self.begin_index = None

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return np.max(self.sigmas)

        return (np.max(self.sigmas) ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increae 1 after each scheduler step.
        """
        return self._step_index

    def _init_step_index(self, timestep: Union[float, np.ndarray]):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self.begin_index

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep
    def index_for_timestep(
        self,
        timestep: Union[float, np.ndarray],
        schedule_timesteps: Union[None, float, np.ndarray]=None,
    ):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0,
                self.config.num_train_timesteps - 1,
                num_inference_steps,
                dtype=np.float32,
            )[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(np.float32)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(self.config.num_train_timesteps, 0, -step_ratio))
                .round()
                .copy()
                .astype(np.float32)
            )
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        self.timesteps = timesteps
        self.sigmas = np.concatenate((sigmas, np.zeros(1)), axis=0)
        self._step_index = None
        self.begin_index = None
        

    def scale_model_input(
        self,
        sample: np.ndarray,
        timestep: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`np.ndarray`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.
        Returns:
            `np.ndarray`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def step(
        self,
        model_output: np.ndarray,
        timestep: Union[float, np.ndarray],
        sample: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            np.ndarray
        """
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = float("inf")
        s_noise = 1.0

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = np.random.randn(*model_output.shape)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample

    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray,
    ):
        schedule_timesteps = self.timesteps
        sigmas = self.sigmas

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement setbegin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called bevore first denoising step to create inital latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]
        sigma = sigmas[step_indices].flatten()

        while len(sigma.shape) < len(original_samples.shape):
            sigma = np.expand_dims(sigma, axis=-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples
