import math
import numpy as np
import math
from typing import Optional, Union

DYNAMIC_THRESHOLDING_RATIO = 0.995
SAMPLE_MAX_VALUE = 1.0
NUM_TRAIN_TIMESTEPS = 1000
PREDICTION_TYPE="epsilon"


class LCMSchedulerOutput:
    """
    Output class for the scheduler"s `step` function output.
    Args:
        prev_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of the previous timestep. `prev_sample` should be used as the next model input in the
            denoising loop.
        pred_original_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    def __init__(self, prev_sample: np.ndarray, denoised: Optional[np.ndarray] = None):
        self.prev_sample = prev_sample
        self.denoised = denoised
    

    def betas_for_alpha_bar(
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
    ):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].
        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.
        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                         prevent singularities.
            alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                         Choose from `cosine` or `exp`
        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """
        if alpha_transform_type == "cosine":
            def alpha_bar_fn(t):
                return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        elif alpha_transform_type == "exp":
            def alpha_bar_fn(t):
                return np.exp(t * -12.0)
        else:
            raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return np.array(betas, dtype=np.float32)
        
        
        
    def rescale_zero_terminal_snr(betas):
        """
        Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        Args:
            betas (`np.ndarray`):
                the betas that the scheduler is being initialized with.
        Returns:
            `np.ndarray`: rescaled betas with zero terminal SNR
        """
        # Convert betas to alphas_bar_sqrt
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_bar_sqrt = np.sqrt(alphas_cumprod)

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
        alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
        alphas = np.concatenate([alphas_bar[:1], alphas])
        betas = 1 - alphas

        return betas
        


class LCMScheduler:
    # ... (other code remains the same)
   
    def __init__( self, num_train_timesteps = 1000, beta_start = 0.0001, beta_end = 0.02,
        beta_schedule = "linear", trained_betas = None, clip_sample = True, set_alpha_to_one = True,
        steps_offset: int = 0, prediction_type = "epsilon", thresholding = False, dynamic_thresholding_ratio = 0.995,
        clip_sample_range = 1.0, sample_max_value = 1.0, timestep_spacing = "leading", rescale_betas_zero_snr = False):
        
        if trained_betas is not None:
            self.betas = np.array(trained_betas, dtype=np.float32)
        elif beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = (np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2)
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)  # Assuming the function is available
        else:
            raise NotImplementedError(str(beta_schedule)+" does is not implemented for ", self.__class__)

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

        self.final_alpha_cumprod = np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.init_noise_sigma = 1.0

        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].astype(np.int64)
        self.prediction_type=prediction_type
        
    def scale_model_input(self, sample: np.ndarray, timestep: Optional[int] = None) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`np.ndarray`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.
        Returns:
            `np.ndarray`:
                A scaled input sample.
        """
        return sample
        
        
    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
        
        
    def _threshold_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (np.float32, np.float64):
            sample = sample.astype(np.float32)  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = np.abs(sample)  

        s = np.percentile(abs_sample, DYNAMIC_THRESHOLDING_RATIO * 100, axis=1)
        s = np.clip(s, a_min=1, a_max= SAMPLE_MAX_VALUE)
        
        s = s[:, np.newaxis]  
        sample = np.clip(sample, -s, s) / s  

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.astype(dtype)

        return sample
        
        
        
    def set_timesteps(self, num_inference_steps: int, lcm_origin_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > NUM_TRAIN_TIMESTEPS:
            raise ValueError(
                str(num_inference_steps)+ " cannot be larger than `self.config.train_timesteps`:"
                " "+str(NUM_TRAIN_TIMESTEPS)+" as the unet model trained with this scheduler can only handle"
                " maximal "+str(NUM_TRAIN_TIMESTEPS)+" timesteps."
            )

        self.num_inference_steps = num_inference_steps
        
        # LCM Timesteps Setting:  # Linear Spacing
        c = NUM_TRAIN_TIMESTEPS // lcm_origin_steps
        lcm_origin_timesteps = np.arange(1, lcm_origin_steps + 1) * c - 1   # LCM Training  Steps Schedule
        skipping_step = len(lcm_origin_timesteps) // num_inference_steps
        timesteps = lcm_origin_timesteps[::-skipping_step][:num_inference_steps]           # LCM Inference Steps Schedule
        
        self.timesteps = timesteps.copy().astype(np.int64)
        
    def get_scalings_for_boundary_condition_discrete(self, t):
        self.sigma_data = 0.5  # Default: 0.5
        
        t_np = np.array(t, dtype=np.float32)
        sigma_data_np = np.array(self.sigma_data, dtype=np.float32)
        
        # By dividing 0.1: This is almost a delta function at t=0.
        c_skip = (sigma_data_np ** 2) / ((t_np / 0.1) ** 2 + sigma_data_np ** 2)
        c_out = ((t_np / 0.1) / ((t_np / 0.1) ** 2 + sigma_data_np ** 2) ** 0.5)
        
        return c_skip, c_out
        
        
        
    def step(
        self,
        model_output,
        timeindex,
        timestep,
        sample,
        eta=0.0,
        use_clipped_model_output=False,
        generator=None,
        variance_noise=None,
        return_dict=True,
    ):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is None, you need to run "+str(set_timesteps)+" after creating the scheduler"
            )
        
        # 1. get previous step value
        prev_timeindex = timeindex + 1
        if prev_timeindex < len(self.timesteps):
            prev_timestep = self.timesteps[prev_timeindex]
        else:
            prev_timestep = timestep
        
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        
        # 4. Different Parameterization:
        parameterization = self.prediction_type
        
        if parameterization == "epsilon":           # noise-prediction
            pred_x0 = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
            
        elif parameterization == "sample":          # x-prediction
            pred_x0 = model_output
            
        elif parameterization == "v_prediction":    # v-prediction
            pred_x0 = np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
        
        # 4. Denoise model output using boundary conditions
        denoised = c_out * pred_x0 + c_skip * sample
        
        # 5. Sample z ~ N(0, I), For MultiStep Inference
        # Noise is not used for one-step sampling.
        if len(self.timesteps) > 1:
            noise = np.random.randn(*model_output.shape)
            prev_sample = np.sqrt(alpha_prod_t_prev) * denoised + np.sqrt(beta_prod_t_prev) * noise
        else:
            prev_sample = denoised
        
        if not return_dict:
            return (prev_sample, denoised)
        
        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)
        
        
    def add_noise(
        self,
        original_samples,
        noise,
        timesteps,
    ):
        
        alphas_cumprod = self.alphas_cumprod.astype(original_samples.dtype)
        timesteps = timesteps.astype(original_samples.dtype)

        sqrt_alpha_prod = np.sqrt(alphas_cumprod[timesteps])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = np.expand_dims(sqrt_alpha_prod, axis=-1)

        sqrt_one_minus_alpha_prod = np.sqrt(1 - alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = np.expand_dims(sqrt_one_minus_alpha_prod, axis=-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
        
    def get_velocity(
        self, sample, noise, timesteps
    ):
        
        alphas_cumprod = self.alphas_cumprod.astype(sample.dtype)
        timesteps = timesteps.astype(sample.dtype)

        sqrt_alpha_prod = np.sqrt(alphas_cumprod[timesteps])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = np.expand_dims(sqrt_alpha_prod, axis=-1)

        sqrt_one_minus_alpha_prod = np.sqrt(1 - alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = np.expand_dims(sqrt_one_minus_alpha_prod, axis=-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
        
    def __len__(self):
        return self.num_train_timesteps
