import os
import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from image_utils import normalize_image  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
# logger
from logging import getLogger  # noqa

from constants import alphas_cumprod
from ddpm_utils import get_fold_unfold

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_FST_ENC_PATH = 'first_stage_encode.onnx'
MODEL_FST_ENC_PATH = 'first_stage_encode.onnx.prototxt'
# WEIGHT_DFSN_PATH = 'diffusion_model.onnx'
# MODEL_DFSN_PATH = 'diffusion_model.onnx.prototxt'
# WEIGHT_AUTO_ENC_PATH = 'autoencoder.onnx'
# MODEL_AUTO_ENC_PATH = 'autoencoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/latent-diffusion-superresolution/'

IMAGE_PATH = 'custom_fox.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Latent Diffusion', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    "--ddim_steps", type=int, default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--ddim_eta", type=float, default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
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
ddim_num_steps = args.ddim_steps
ddpm_num_timesteps = 1000
ddim_timesteps = make_ddim_timesteps(
    ddim_num_steps, ddpm_num_timesteps)

"""
ddim sampling parameters
"""
ddim_eta = args.ddim_eta
ddim_sigmas, ddim_alphas, ddim_alphas_prev = \
    make_ddim_sampling_parameters(
        alphacums=alphas_cumprod,
        ddim_timesteps=ddim_timesteps,
        eta=ddim_eta)

ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)


def preprocess(img):
    im_h, im_w, _ = img.shape

    up_f = 4
    oh, ow = up_f * im_h, up_f * im_w

    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
    img = normalize_image(img, normalize_type='255')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def ddim_sampling(
        models,
        cond, shape):
    img = np.random.randn(shape[0] * shape[1] * shape[2] * shape[3]).reshape(shape)
    img = img.astype(np.float32)

    timesteps = ddim_timesteps
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    logger.info(f"Running DDIM Sampling with {total_steps} timesteps")

    try:
        from tqdm import tqdm
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    except ModuleNotFoundError:
        def iter_func(a):
            for i, x in enumerate(a):
                print("DDIM Sampler: %s/%s" % (i + 1, len(a)))
                yield x

        iterator = iter_func(time_range)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = np.full((shape[0],), step, dtype=np.int64)

        img, pred_x0 = p_sample_ddim(
            models,
            img, cond, ts,
            index=index,
        )
        img = img.astype(np.float32)

    return img


# ddim
def p_sample_ddim(
        models, x, c, t, index,
        temperature=1):
    e_t = apply_model(models, x, t, c)

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


def encode_first_stage(models, x):
    ks = (128, 128)
    stride = (64, 64)
    df = 4

    bs, nc, h, w = x.shape

    fold, unfold, weighting = get_fold_unfold(x, ks, stride, df=df)
    z, o_shape, _ = unfold(x)  # (bn, nc * prod(**ks), L)

    # Reshape to img shape
    z = z.reshape((bs, -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L)

    first_stage_encode = models['first_stage_encode']
    outputs = []
    for i in range(z.shape[-1]):
        x = z[:, :, :, :, i]
        if not args.onnx:
            output = first_stage_encode.predict([x])
        else:
            output = first_stage_encode.run(None, {'x': x})
        outputs.append(output[0])

    o = np.stack(outputs, axis=-1)
    o = o * weighting

    # Reverse reshape to img shape
    o = o.reshape((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
    decoded = fold(o, I_shape=(1, 3, h // df, w // df), O_shape=o_shape)

    normalization = fold(weighting, I_shape=(1, 1, h // df, w // df), O_shape=o_shape)
    normalization = normalization.reshape(1, 1, h // df, w // df)
    decoded = decoded / normalization

    return decoded


# ddpm
def apply_model(models, x, t, cond):
    diffusion_model = models["diffusion_model"]

    xc = np.concatenate([x, cond], axis=1)
    if not args.onnx:
        output = diffusion_model.predict([xc, t])
    else:
        output = diffusion_model.run(None, {'xc': xc, 't': t})
    x_recon = output[0]

    return x_recon


# decoder
def decode_first_stage(models, z):
    scale_factor = 1.0
    z = z / scale_factor

    autoencoder = models['autoencoder']
    if not args.onnx:
        output = autoencoder.predict([z])
    else:
        output = autoencoder.run(None, {'z': z})
    dec = output[0]

    return dec


def predict(models, img):
    img = img[:, :, ::-1]  # BGR -> RGB

    img = preprocess(img)

    encode_first_stage(models, img)

    x_samples_ddim = decode_first_stage(models, samples)

    return None


def recognize_from_image(models):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # mask = np.array(Image.open(mask_path).convert("L"))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(models, img)

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, output)

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(WEIGHT_FST_ENC_PATH, MODEL_FST_ENC_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_DFSN_PATH, MODEL_DFSN_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_AUTO_ENC_PATH, MODEL_AUTO_ENC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        logger.info("This model requires 10GB or more memory.")
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=False)
        first_stage_encode = ailia.Net(
            MODEL_FST_ENC_PATH, WEIGHT_FST_ENC_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        first_stage_encode = onnxruntime.InferenceSession(WEIGHT_FST_ENC_PATH)

    models = dict(
        first_stage_encode=first_stage_encode,
    )
    recognize_from_image(models)


if __name__ == '__main__':
    main()
