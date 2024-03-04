import warnings

import numpy as np
from tqdm.auto import trange

from constants import log_sigmas

# be rewritten later
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


def sigma_to_t(sigma):
    log_sigma = np.log(sigma)
    dists = log_sigma - log_sigmas[:, None]

    return np.abs(dists).argmin(axis=0).reshape(sigma.shape)


def get_scalings(sigma):
    sigma_data = 1.0
    c_out = -sigma
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_out, c_in


def get_eps(models, input, sigma, cond):
    def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        return x[(...,) + (None,) * dims_to_append]

    c_out, c_in = [append_dims(x, input.ndim) for x in get_scalings(sigma)]
    eps = apply_model(models, input * c_in, sigma_to_t(sigma), cond)

    return input + eps * c_out


def CFGDenoiser(
        models, x, sigma, uncond, cond, cond_scale):
    batch_size = 1
    tensor = np.expand_dims(cond, axis=0)

    x_in = np.concatenate([x, x])
    sigma_in = np.concatenate([sigma, sigma])

    x_out = np.zeros_like(x_in)
    batch_size = batch_size * 2
    for batch_offset in range(0, tensor.shape[0], batch_size):
        a = batch_offset
        b = min(a + batch_size, tensor.shape[0])
        c_crossattn = tensor[a:b]

        x_out[a:b] = get_eps(models, x_in[a:b], sigma_in[a:b], cond=c_crossattn[0])

    x_out[-uncond.shape[0]:] = get_eps(
        models, x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond=uncond)

    denoised_uncond = x_out[-uncond.shape[0]:]
    denoised = np.copy(denoised_uncond)
    denoised[0] += (x_out[0] - denoised_uncond[0]) * cond_scale

    return denoised


def sample(
        models, x, conditioning,
        unconditional_conditioning,
        sampler, steps=20, cfg_scale=1.0,
        **kwargs):
    sigmas = get_sigmas(steps)

    x = x * sigmas[0]

    extra_params_kwargs = {}
    extra_params_kwargs['sigmas'] = sigmas

    samples = sampler(models, x, extra_args={
        'cond': conditioning,
        'uncond': unconditional_conditioning,
        'cond_scale': cfg_scale,
    }, **extra_params_kwargs)

    return samples


def sample_dpmpp_2m(models, x, sigmas, extra_args=None):
    """DPM-Solver++(2M)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        extra_args = {} if extra_args is None else extra_args
        s_in = np.array([1] * x.shape[0])
        sigma_fn = lambda t: np.exp(-t)
        t_fn = lambda sigma: -np.log(sigma)

        old_denoised = None
        for i in trange(len(sigmas) - 1):
            denoised = CFGDenoiser(models, x, sigmas[i] * s_in, **extra_args)

            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - np.expm1(-h) * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - np.expm1(-h) * denoised_d
            old_denoised = denoised

    return x
