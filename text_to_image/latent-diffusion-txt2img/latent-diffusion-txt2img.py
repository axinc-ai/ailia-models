import sys
import time

import numpy as np
import cv2
from PIL import Image

from transformers import BertTokenizerFast

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# from image_utils import load_image, normalize_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from constants import alphas_cumprod

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_TRANS_EMB_PATH = 'transformer_emb.onnx'
MODEL_TRANS_EMB_PATH = 'transformer_emb.onnx.prototxt'
WEIGHT_TRANS_ATTN_PATH = 'transformer_attn.onnx'
MODEL_TRANS_ATTN_PATH = 'transformer_attn.onnx.prototxt'
WEIGHT_DFSN_EMB_PATH = 'diffusion_emb.onnx'
MODEL_DFSN_EMB_PATH = 'diffusion_emb.onnx.prototxt'
WEIGHT_DFSN_MID_PATH = 'diffusion_mid.onnx'
MODEL_DFSN_MID_PATH = 'diffusion_mid.onnx.prototxt'
WEIGHT_DFSN_OUT_PATH = 'diffusion_out.onnx'
MODEL_DFSN_OUT_PATH = 'diffusion_out.onnx.prototxt'
WEIGHT_AUTO_ENC_PATH = 'autoencoder.onnx'
MODEL_AUTO_ENC_PATH = 'autoencoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Latent Diffusion', None, None
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def make_ddim_timesteps(num_ddim_timesteps, num_ddpm_timesteps):
    c = num_ddpm_timesteps // num_ddim_timesteps
    ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1

    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

    return sigmas, alphas, alphas_prev


# ======================
# Main functions
# ======================

"""
ddim_timesteps
"""
ddim_num_steps = 50
ddpm_num_timesteps = 1000
ddim_timesteps = make_ddim_timesteps(
    ddim_num_steps, ddpm_num_timesteps)

"""
ddim sampling parameters
"""
ddim_eta = 0.0
ddim_sigmas, ddim_alphas, ddim_alphas_prev = \
    make_ddim_sampling_parameters(
        alphacums=alphas_cumprod,
        ddim_timesteps=ddim_timesteps,
        eta=ddim_eta)

ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)


# encoder
class BERTEmbedder:
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, transformer_emb, transformer_attn, max_length=77):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        self.transformer_emb = transformer_emb
        self.transformer_attn = transformer_attn

    def encode(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]
        tokens = tokens.numpy()

        if not args.onnx:
            output = self.transformer_emb.predict([tokens])
        else:
            output = self.transformer_emb.run(None, {'x': tokens})
        x = output[0]

        if not args.onnx:
            output = self.transformer_attn.predict([x])
        else:
            output = self.transformer_attn.run(None, {'x': x})
        z = output[0]

        return z


def ddim_sampling(
        models,
        cond, shape,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None):
    img = np.random.randn(shape[0] * shape[1] * shape[2] * shape[3]).reshape(shape)
    img = np.load("img.npy")

    timesteps = ddim_timesteps
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    logger.info(f"Running DDIM Sampling with {total_steps} timesteps")

    from tqdm import tqdm
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = np.full((shape[0],), step, dtype=np.int64)

        img, pred_x0 = p_sample_ddim(
            models,
            img, cond, ts,
            index=index,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        img = img.astype(np.float32)

    return img


# ddim
def p_sample_ddim(
        models, x, c, t, index,
        temperature=1.,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None):
    x_in = np.concatenate([x] * 2)
    t_in = np.concatenate([t] * 2)
    c_in = np.concatenate([unconditional_conditioning, c])

    x_recon = apply_model(models, x_in, t_in, c_in)
    e_t_uncond, e_t = np.split(x_recon, 2)

    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

    alphas = ddim_alphas
    alphas_prev = ddim_alphas_prev
    sqrt_one_minus_alphas = ddim_sqrt_one_minus_alphas
    sigmas = ddim_sigmas

    # select parameters corresponding to the currently considered timestep
    b, *_ = x.shape
    a_t = np.full((b, 1, 1, 1), alphas[index])
    a_prev = np.full((b, 1, 1, 1), alphas_prev[index])
    sigma_t = np.full((b, 1, 1, 1), sigmas[index])
    sqrt_one_minus_at = np.full((b, 1, 1, 1), sqrt_one_minus_alphas[index])

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / np.sqrt(a_t)

    # direction pointing to x_t
    dir_xt = np.sqrt(1. - a_prev - sigma_t ** 2) * e_t

    noise = sigma_t * np.random.randn(x.size).reshape(x.shape) * temperature
    x_prev = np.sqrt(a_prev) * pred_x0 + dir_xt + noise

    return x_prev, pred_x0


# ddpm
def apply_model(models, x, t, cc):
    diffusion_emb = models["diffusion_emb"]
    diffusion_mid = models["diffusion_mid"]
    diffusion_out = models["diffusion_out"]

    if not args.onnx:
        output = diffusion_emb.predict([x, t, cc])
    else:
        output = diffusion_emb.run(None, {'x': x, 'timesteps': t, 'context': cc})
    h, emb, *hs = output

    if not args.onnx:
        output = diffusion_mid.predict([h, emb, cc, *hs[6:]])
    else:
        output = diffusion_mid.run(None, {
            'h': h, 'emb': emb, 'context': cc,
            'h6': hs[6], 'h7': hs[7], 'h8': hs[8],
            'h9': hs[9], 'h10': hs[10], 'h11': hs[11],
        })
    h = output[0]

    if not args.onnx:
        output = diffusion_out.predict([h, emb, cc, *hs[:6]])
    else:
        output = diffusion_out.run(None, {
            'h': h, 'emb': emb, 'context': cc,
            'h0': hs[0], 'h1': hs[1], 'h2': hs[2],
            'h3': hs[3], 'h4': hs[4], 'h5': hs[5],
        })
    out = output[0]

    return out


# decoder
def decode_first_stage(models, z):
    scale_factor = 0.18215
    z = z / scale_factor

    autoencoder = models['autoencoder']
    if not args.onnx:
        output = autoencoder.predict([z])
    else:
        output = autoencoder.run(None, {'input': z})
    dec = output[0]

    return dec


def recognize_from_text(models):
    n_samples = 4
    n_iter = 1
    scale = 5.0
    H = W = 256

    transformer_emb = models['transformer_emb']
    transformer_attn = models['transformer_attn']
    cond_stage_model = BERTEmbedder(transformer_emb, transformer_attn)

    uc = None
    if scale != 1.0:
        uc = cond_stage_model.encode([""] * n_samples)

    prompt = "korean girl"
    for _ in range(n_iter):
        c = cond_stage_model.encode([prompt] * n_samples)
        shape = [n_samples, 4, H // 8, W // 8]

        samples = ddim_sampling(
            models, c, shape,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc)
        # samples = np.load("samples.npy")
        # print("samples_ddim---", samples_ddim)
        # print("samples_ddim---", samples_ddim.shape)

        x_samples_ddim = decode_first_stage(models, samples)
        x_samples_ddim = np.clip((x_samples_ddim + 1.0) / 2.0, a_min=0.0, a_max=1.0)

        base_count = 0
        for x_sample in x_samples_ddim:
            x_sample = x_sample.transpose(1, 2, 0)  # CHW -> HWC
            x_sample = x_sample * 255
            img = x_sample.astype(np.uint8)

            Image.fromarray(img).save(f"{base_count:04}.png")
            base_count += 1

    logger.info('Script finished successfully.')


def main():
    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=False)
        transformer_emb = ailia.Net(
            MODEL_TRANS_EMB_PATH, WEIGHT_TRANS_EMB_PATH, env_id=env_id, memory_mode=memory_mode)
        transformer_attn = ailia.Net(
            MODEL_TRANS_ATTN_PATH, WEIGHT_TRANS_ATTN_PATH, env_id=env_id, memory_mode=memory_mode)
        diffusion_emb = ailia.Net \
            (MODEL_DFSN_EMB_PATH, WEIGHT_DFSN_EMB_PATH, env_id=env_id, memory_mode=memory_mode)
        diffusion_mid = ailia.Net(
            MODEL_DFSN_MID_PATH, WEIGHT_DFSN_MID_PATH, env_id=env_id, memory_mode=memory_mode)
        diffusion_out = ailia.Net(
            MODEL_DFSN_OUT_PATH, WEIGHT_DFSN_OUT_PATH, env_id=env_id, memory_mode=memory_mode)
        autoencoder = ailia.Net(
            MODEL_AUTO_ENC_PATH, WEIGHT_AUTO_ENC_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        transformer_emb = onnxruntime.InferenceSession(WEIGHT_TRANS_EMB_PATH)
        transformer_attn = onnxruntime.InferenceSession(WEIGHT_TRANS_ATTN_PATH)
        diffusion_emb = onnxruntime.InferenceSession(WEIGHT_DFSN_EMB_PATH)
        diffusion_mid = onnxruntime.InferenceSession(WEIGHT_DFSN_MID_PATH)
        diffusion_out = onnxruntime.InferenceSession(WEIGHT_DFSN_OUT_PATH)
        autoencoder = onnxruntime.InferenceSession(WEIGHT_AUTO_ENC_PATH)

    models = dict(
        transformer_emb=transformer_emb,
        transformer_attn=transformer_attn,
        diffusion_emb=diffusion_emb,
        diffusion_mid=diffusion_mid,
        diffusion_out=diffusion_out,
        autoencoder=autoencoder,
    )
    recognize_from_text(models)


if __name__ == '__main__':
    main()
