import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_FFHQ_ENC_PATH = "ffhq_encoder.onnx"
MODEL_FFHQ_ENC_PATH = "ffhq_encoder.onnx.prototxt"
WEIGHT_FFHQ_DEC_PATH = "ffhq_decoder.onnx"
MODEL_FFHQ_DEC_PATH = "ffhq_decoder.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/e4e/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Encoder for StyleGAN Image Manipulation', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--age_factor', default=None, type=int,
    help='InterFaceGAN: age-factor'
)
parser.add_argument(
    '--age_range', default=None, type=int, nargs='+',
    help='InterFaceGAN: age-range'
)
parser.add_argument(
    '--eye_openness', default=None, type=int,
    help='GANSpace: eye_openness: The larger the value, the more closes.'
)
parser.add_argument(
    '--smile', default=None, type=int,
    help='GANSpace: smile: The smaller the value, the more smile.'
)
parser.add_argument(
    '--trimmed_beard', default=None, type=int,
    help='GANSpace: trimmed_beard'
)
parser.add_argument(
    '--white_hair', default=None, type=int,
    help='GANSpace: white_hair: The smaller the value, the more white.'
)
parser.add_argument(
    '--lipstick', default=None, type=int,
    help='GANSpace: lipstick'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Main functions
# ======================

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB

    ow = oh = 256
    # img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))
    img = normalize_image(img, normalize_type='127.5')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR

    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img = img * 255
    img = img.astype(np.uint8)

    return img


def apply_interfacegan(latent, direction, factor=1, factor_range=None):
    edit_latents = []
    if factor_range is not None:
        for f in range(*factor_range):
            edit_latent = latent + f * direction
            edit_latents.append(edit_latent)
    else:
        edit_latents = [latent, latent + factor * direction]

    return edit_latents


def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean']
    lat_comp = pca['comp']
    lat_std = pca['std']
    w_coord = np.sum(w_centered[0].reshape(-1) * lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord) * lat_comp[idx] * lat_std[idx]

    return delta


def apply_ganspace(latents, pca, edit_directions):
    edit_latents = []

    for latent in latents:
        for pca_idx, start, end, strength in edit_directions:
            delta = get_delta(pca, latent, pca_idx, strength)
            delta_padded = np.zeros(latent.shape, dtype=np.float32)
            delta_padded[start:end] += np.repeat(delta, end - start, axis=0)
            edit_latent = latent + delta_padded
            edit_latents.append(np.expand_dims(edit_latent, axis=0))

    return edit_latents


def predict(models, img):
    net_enc = models["enc"]
    net_dec = models["dec"]

    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net_enc.predict([img])
    else:
        output = net_enc.run(None, {'x': img})
    latents = output[0]

    age_factor = args.age_factor
    age_range = args.age_range
    eye_openness = args.eye_openness
    smile = args.smile
    trimmed_beard = args.trimmed_beard
    white_hair = args.white_hair
    lipstick = args.lipstick

    if age_factor or age_range:
        interfacegan_direction = models['interfacegan_direction'] = \
            models.get('interfacegan_direction', np.load("editings/interfacegan_directions/age.npy"))

        if age_range:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor_range=age_range)
        else:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor=age_factor)
    elif eye_openness or smile or trimmed_beard or white_hair or lipstick:
        ganspace_pca = models['ganspace_pca'] = \
            models.get('ganspace_pca', np.load("editings/ganspace_pca/ffhq_pca.npy", allow_pickle=True).item())

        directions = {
            'eye_openness': (54, 7, 8, eye_openness),
            'smile': (46, 4, 5, smile),
            'trimmed_beard': (58, 7, 9, trimmed_beard),
            'white_hair': (57, 7, 10, white_hair),
            'lipstick': (34, 10, 11, lipstick)
        }
        directions = [v for k, v in directions.items() if v[3]]
        edit_latents = apply_ganspace(latents, ganspace_pca, directions)
    else:
        edit_latents = [latents]

    preds = []
    for latent in edit_latents:
        if not args.onnx:
            output = net_dec.predict([latent])
        else:
            output = net_dec.run(None, {'latent': latent})
        pred = output[0]
        preds.append(pred)

    imgs = [post_processing(pred) for pred in preds]
    out_img = np.concatenate(imgs, axis=1)

    return out_img


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
                out_img = predict(models, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out_img = predict(models, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out_img)

    logger.info('Script finished successfully.')


def recognize_from_video(models):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        restored_img = predict(models, img)

        # show
        cv2.imshow('frame', restored_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(restored_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    WEIGHT_ENC_PATH = WEIGHT_FFHQ_ENC_PATH
    MODEL_ENC_PATH = MODEL_FFHQ_ENC_PATH
    WEIGHT_DEC_PATH = WEIGHT_FFHQ_DEC_PATH
    MODEL_DEC_PATH = MODEL_FFHQ_DEC_PATH
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_enc = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
        net_dec = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_enc = onnxruntime.InferenceSession(WEIGHT_ENC_PATH)
        net_dec = onnxruntime.InferenceSession(WEIGHT_DEC_PATH)

    models = {
        "enc": net_enc,
        "dec": net_dec,
    }

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
