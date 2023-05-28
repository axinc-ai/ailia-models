import numpy as np

apply_model = lambda models, x, t, cc: None


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = np.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    return np.concatenate([sigmas, np.array([0])])


def get_sigmas(steps):
    sigma_min, sigma_max = 0.0312652550637722, 14.611639022827148

    sigmas = get_sigmas_karras(
        n=steps, sigma_min=sigma_min, sigma_max=sigma_max)

    return sigmas


def sample(
        models, x, conditioning,
        unconditional_conditioning,
        sampling_func, steps=20, cfg_scale=1.0,
        **kwargs):
    image_conditioning = None
    s_min_uncond = 0

    sigmas = get_sigmas(steps)

    x = x * sigmas[0]

    extra_params_kwargs = {}
    extra_params_kwargs['sigmas'] = sigmas

    samples = sampling_func(models, x, extra_args={
        'cond': conditioning,
        'image_cond': image_conditioning,
        'uncond': unconditional_conditioning,
        'cond_scale': cfg_scale,
        's_min_uncond': s_min_uncond
    }, **extra_params_kwargs)

    return samples




def sample_dpmpp_2m(model, x, sigmas, extra_args=None):
    pass
