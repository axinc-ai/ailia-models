import sys
import time
import math
from logging import getLogger

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = 'unet.onnx'
MODEL_UNET_PATH = 'unet.onnx.prototxt'
WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/marigold/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Marigold', IMAGE_PATH, SAVE_IMAGE_PATH
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

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================

def preprocess(img):
    im_h, im_w, _ = img.shape

    max_edge_resolution = 768
    downscale_factor = min(
        max_edge_resolution / im_w, max_edge_resolution / im_h
    )
    ow = int(im_w * downscale_factor)
    oh = int(im_h * downscale_factor)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))

    img = normalize_image(img, normalize_type='255')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    # img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def single_infer(
        models,
        rgb_in,
        num_inference_steps):
    scheduler = models["scheduler"]
    timesteps = scheduler.set_timesteps(num_inference_steps)


def post_processing(output):
    return None


def predict(models, img):
    img = img[..., ::-1]  # BGR -> RGB
    img = preprocess(img)

    ensemble_size = 10
    imgs = np.stack([img] * ensemble_size)

    bs = ensemble_size // 2
    denoising_steps = 10

    cnt = int(math.ceil(ensemble_size / bs))
    bar = tqdm(
        total=cnt, desc=" " * 2 + "Inference batches", leave=False
    )
    for i in range(cnt):
        batched_img = imgs[i * bs:(i + 1) * bs]
        depth_pred_raw = single_infer(
            models=models,
            rgb_in=batched_img,
            num_inference_steps=denoising_steps,
        )

        bar.update(1)

    # pred = post_processing(output)

    return


def recognize_from_image(models):
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(models, img)

        # res_img = draw_bbox(out)
        res_img = img

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def main():
    env_id = args.env_id

    # initialize
    if not args.onnx:
        unet = ailia.Net(MODEL_UNET_PATH, WEIGHT_UNET_PATH, env_id=env_id)
        vae_decoder = ailia.Net(MODEL_VAE_DECODER_PATH, WEIGHT_VAE_DECODER_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        unet = onnxruntime.InferenceSession(WEIGHT_UNET_PATH, providers=providers)
        vae_decoder = onnxruntime.InferenceSession(WEIGHT_VAE_DECODER_PATH, providers=providers)

    scheduler = df.schedulers.DDIMScheduler.from_config({
        "prediction_type": "v_prediction",
        "clip_sample": False,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "set_alpha_to_one": False,
        "trained_betas": None,
        "steps_offset": 1,
        "beta_end": 0.012,
        "num_train_timesteps": 1000,
    })

    # unet_input = np.load("unet_input.npy")
    # batch_empty_text_embed = np.load("batch_empty_text_embed.npy")
    # t = np.array([901], dtype=np.float32)
    #
    # print("1---", unet_input)
    # print("1---", unet_input.shape)
    # print("2---", t)
    # print("2---", t.shape)
    # print("3---", batch_empty_text_embed)
    # print("3---", batch_empty_text_embed.shape)
    # output = unet.run(None, {
    #     'sample': unet_input, 'timestep': t, 'encoder_hidden_states': batch_empty_text_embed
    # })
    # noise_pred = output[0]
    # print("4---", noise_pred)
    # print("4---", noise_pred.shape)

    # depth_latent = np.load("depth_latent.npy")
    # print("5---", depth_latent)
    # print("5---", depth_latent.shape)
    # output = vae_decoder.run(None, {
    #     'latent_sample': depth_latent
    # })
    # stacked = output[0]
    # print("6---", stacked)
    # print("6---", stacked.shape)

    models = {
        "unet": unet,
        "vae_decoder": vae_decoder,
        "scheduler": scheduler,
    }

    recognize_from_image(models)


if __name__ == '__main__':
    main()
