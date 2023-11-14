import abc
import math
import numpy as np

from tqdm import tqdm
from collections import OrderedDict

class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self,net,onnx, max_sigma, T=100, sample_T=-1, eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self.sample_T = self.T if sample_T < 0 else sample_T
        self.sample_scale = self.T / self.sample_T
        self._initialize(self.max_sigma, self.sample_T, eps)
        self.net = net
        self.onnx = onnx

    def _initialize(self, max_sigma, T, eps=0.01):

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps, dtype=np.float32)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return np.cumsum(thetas, axis=0)

        def get_sigmas(thetas):
            return np.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return np.sqrt(max_sigma**2 * (1 - np.exp(-2 * thetas_cumsum * self.dt)))
            
        thetas = cosine_theta_schedule(T)

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas
        self.sigmas = sigmas
        self.thetas_cumsum = thetas_cumsum
        self.sigma_bars = sigma_bars

        self.mu = 0.
        self.model = None
        self.net = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (np.random.randn(*x.shape).astype(np.float32) * math.sqrt(self.dt))

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)


    def score_fn(self, x,mu, t, scale, kwargs1,kwargs2):
        
        tmp = np.expand_dims(t * scale, 0)
        if self.onnx:
            name = self.net.get_inputs()
            noise = self.net.run([],{name[0].name:x.astype(np.float32),
                                     name[1].name:mu.astype(np.float32),
                                     name[2].name:tmp.astype(np.float32),
                                     name[3].name:kwargs1.astype(np.float32),
                                     name[4].name:kwargs2.astype(np.float32)})[0]
        else:
            noise = self.net.run((x.astype(np.float32),
                                  mu.astype(np.float32),
                                  tmp.astype(np.float32),
                                  kwargs1.astype(np.float32),
                                  kwargs2.astype(np.float32)))[0]
 
        return self.get_score_from_noise(noise, t)


    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        T = self.sample_T if T < 0 else T
        x = xt

        for t in reversed(range(1, T + 1)):
            score = self.score_fn(x,
                                  self.mu,
                                  t,
                                  self.sample_scale,
                                  kwargs['text_context'],
                                  kwargs['image_context'])
            x = self.reverse_sde_step(x, score, t)
        return x

    def noise_state(self, tensor):
        return tensor + np.random.randn(*tensor.shape).astype(np.float32) * self.max_sigma



def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = np.squeeze(tensor)
    tensor = np.clip(tensor,*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.ndim
    if n_dim == 3:
        img_np = tensor
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        pass
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

class DenoisingModel():
    def __init__(self):
        pass

    def feed_data(self, state, LQ, text_context=None, image_context=None):
        self.state = state    # noisy_state
        self.condition = LQ   # LQ

        self.text_context = text_context
        self.image_context = image_context

    def run(self, sde=None, save_states=False):
        sde.set_mu(self.condition)

        self.output = sde.reverse_sde(self.state, save_states=save_states, text_context=self.text_context, image_context=self.image_context)

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition[0]
        out_dict["Output"] = self.output[0]
        return out_dict


