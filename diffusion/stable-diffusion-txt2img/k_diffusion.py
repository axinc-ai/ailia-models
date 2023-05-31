import numpy as np
from tqdm.auto import trange

from constants import log_sigmas

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
    sigma_data = 0
    c_out = -sigma
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_out, c_in


def get_eps(models, x_noisy, t, cond):
    # x_noisy = np.load("x_noisy.npy")
    x_recon = apply_model(models, x_noisy, t, cond)
    return x_recon


def inner_model(models, input, sigma, cond):
    def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        return x[(...,) + (None,) * dims_to_append]

    c_out, c_in = [append_dims(x, input.ndim) for x in get_scalings(sigma)]
    eps = get_eps(models, input * c_in, sigma_to_t(sigma), cond)

    return input + eps * c_out


def CFGDenoiser(
        models, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
    denoiser_params = CFGDenoiserParams(
        x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps,
        tensor, uncond)
    cfg_denoiser_callback(denoiser_params)
    x_in = denoiser_params.x
    image_cond_in = denoiser_params.image_cond
    sigma_in = denoiser_params.sigma
    tensor = denoiser_params.text_cond
    uncond = denoiser_params.text_uncond
    skip_uncond = False

    x_out = torch.zeros_like(x_in)
    batch_size = batch_size * 2 if shared.batch_cond_uncond else batch_size
    for batch_offset in range(0, tensor.shape[0], batch_size):
        a = batch_offset
        b = min(a + batch_size, tensor.shape[0])

        if not is_edit_model:
            c_crossattn = [tensor[a:b]]
        else:
            c_crossattn = torch.cat([tensor[a:b]], uncond)

        # x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=make_condition_dict(c_crossattn, image_cond_in[a:b]))
        r = inner_model(models, x_in[a:b], sigma_in[a:b], cond=make_condition_dict(c_crossattn, image_cond_in[a:b]))
        print("r---", r)
        print("r---", r.shape)
        1 / 0
        x_out[a:b] = r


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


def sample_dpmpp_2m(models, x, sigmas, extra_args=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = np.array([1] * x.shape[0])
    sigma_fn = lambda t: np.exp(-t)
    t_fn = lambda sigma: -np.log(sigma)
    old_denoised = None

    for i in trange(len(sigmas) - 1):
        denoised = CFGDenoiser(models, x, sigmas[i] * s_in, **extra_args)
        print("denoised---", denoised)
        print("denoised---", denoised.shape)
        1 / 0

        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised

    return x
