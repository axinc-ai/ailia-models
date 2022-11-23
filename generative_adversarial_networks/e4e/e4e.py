import sys
import os
import time

import numpy as np
import cv2

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
WEIGHT_CAR_ENC_PATH = "cars_encoder.onnx"
MODEL_CAR_ENC_PATH = "cars_encoder.onnx.prototxt"
WEIGHT_CAR_DEC_PATH = "cars_decoder.onnx"
MODEL_CAR_DEC_PATH = "cars_decoder.onnx.prototxt"
WEIGHT_HORSE_ENC_PATH = "horse_encoder.onnx"
MODEL_HORSE_ENC_PATH = "horse_encoder.onnx.prototxt"
WEIGHT_HORSE_DEC_PATH = "horse_decoder.onnx"
MODEL_HORSE_DEC_PATH = "horse_decoder.onnx.prototxt"
WEIGHT_CHURCH_ENC_PATH = "church_encoder.onnx"
MODEL_CHURCH_ENC_PATH = "church_encoder.onnx.prototxt"
WEIGHT_CHURCH_DEC_PATH = "church_decoder.onnx"
MODEL_CHURCH_DEC_PATH = "church_decoder.onnx.prototxt"
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/e4e/'

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

p = os.path.os.path.dirname(os.path.abspath(__file__))
FFHQ_PCA = os.path.join(p, 'editings/ganspace_pca/ffhq_pca.npy')
CARS_PCA = os.path.join(p, 'editings/ganspace_pca/cars_pca.npy')

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Encoder for StyleGAN Image Manipulation', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-m', '--model_type', default='ffhq', choices=('ffhq', 'car', 'horse', 'church'),
    help='model type'
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
    '--smile_factor', default=None, type=int,
    help='InterFaceGAN: smile-factor'
)
parser.add_argument(
    '--smile_range', default=None, type=int, nargs='+',
    help='InterFaceGAN: smile-range'
)
parser.add_argument(
    '--pose_factor', default=None, type=int,
    help='InterFaceGAN: pose-factor'
)
parser.add_argument(
    '--pose_range', default=None, type=int, nargs='+',
    help='InterFaceGAN: pose-range'
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
    help='GANSpace: lipstick: The larger the value, the darker the color.'
)
parser.add_argument(
    '--car_view1', default=None, type=int,
    help='GANSpace: Viewpoint I'
)
parser.add_argument(
    '--car_view2', default=None, type=int,
    help='GANSpace: Viewpoint II'
)
parser.add_argument(
    '--car_cube', default=None, type=int,
    help='GANSpace: Cube'
)
parser.add_argument(
    '--car_color', default=None, type=int,
    help='GANSpace: Color'
)
parser.add_argument(
    '--car_grass', default=None, type=int,
    help='GANSpace: Grass'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

model_type = args.model_type


# ======================
# Secondaty Functions
# ======================

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


def edit_ffhq(latents, models):
    age_factor = args.age_factor
    age_range = args.age_range
    smile_factor = args.smile_factor
    smile_range = args.smile_range
    pose_factor = args.pose_factor
    pose_range = args.pose_range
    eye_openness = args.eye_openness
    smile = args.smile
    trimmed_beard = args.trimmed_beard
    white_hair = args.white_hair
    lipstick = args.lipstick

    edit_latents = [latents]
    if age_factor or age_range:
        interfacegan_direction = models['interfacegan_direction'] = \
            models.get('interfacegan_direction', np.load("editings/interfacegan_directions/age.npy"))
        if age_range:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor_range=age_range)
        else:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor=age_factor)
    elif smile_factor or smile_range:
        interfacegan_direction = models['interfacegan_direction'] = \
            models.get('interfacegan_direction', np.load("editings/interfacegan_directions/smile.npy"))
        if smile_range:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor_range=smile_range)
        else:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor=smile_factor)
    elif pose_factor or pose_range:
        interfacegan_direction = models['interfacegan_direction'] = \
            models.get('interfacegan_direction', np.load("editings/interfacegan_directions/pose.npy"))
        if pose_range:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor_range=pose_range)
        else:
            edit_latents = apply_interfacegan(latents, interfacegan_direction, factor=pose_factor)
    elif eye_openness or smile or trimmed_beard or white_hair or lipstick:
        ganspace_pca = models['ganspace_pca'] = \
            models.get('ganspace_pca', np.load(FFHQ_PCA, allow_pickle=True).item())

        directions = {
            'eye_openness': (54, 7, 8, eye_openness),
            'smile': (46, 4, 5, smile),
            'trimmed_beard': (58, 7, 9, trimmed_beard),
            'white_hair': (57, 7, 10, white_hair),
            'lipstick': (34, 10, 11, lipstick)
        }
        directions = [v for k, v in directions.items() if v[3]]
        edit_latents = apply_ganspace(latents, ganspace_pca, directions)

    return edit_latents


def edit_cars(latents, models):
    car_view1 = args.car_view1
    car_view2 = args.car_view2
    car_cube = args.car_cube
    car_color = args.car_color
    car_grass = args.car_grass

    edit_latents = [latents]
    if car_view1 or car_view2 or car_cube or car_color or car_grass:
        ganspace_pca = models['ganspace_pca'] = \
            models.get('ganspace_pca', np.load(CARS_PCA, allow_pickle=True).item())

        directions = {
            "viewpoint_1": (0, 0, 5, car_view1),
            "viewpoint_2": (0, 0, 5, car_view2),
            "cube": (16, 3, 6, car_cube),
            "color": (22, 9, 11, car_color),
            "grass": (41, 9, 11, car_grass),
        }
        directions = [v for k, v in directions.items() if v[3]]
        edit_latents = apply_ganspace(latents, ganspace_pca, directions)

    return edit_latents


# ======================
# Main functions
# ======================

def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RGB

    ow = oh = 256
    if model_type == 'car':
        oh = 192

    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
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

    edit_latents = [latents]
    if model_type == 'ffhq':
        edit_latents = edit_ffhq(latents, models)
    elif model_type == 'car':
        edit_latents = edit_cars(latents, models)

    preds = []
    for latent in edit_latents:
        if not args.onnx:
            output = net_dec.predict([latent])
        else:
            output = net_dec.run(None, {'latent': latent})
        pred = output[0]

        if model_type == 'car':
            pred = pred[:, :, 64:448, :]  # 512x512 -> 384x512

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
    dic_model = {
        'ffhq': {
            'enc': (WEIGHT_FFHQ_ENC_PATH, MODEL_FFHQ_ENC_PATH),
            'dec': (WEIGHT_FFHQ_DEC_PATH, MODEL_FFHQ_DEC_PATH)
        },
        'car': {
            'enc': (WEIGHT_CAR_ENC_PATH, MODEL_CAR_ENC_PATH),
            'dec': (WEIGHT_CAR_DEC_PATH, MODEL_CAR_DEC_PATH)
        },
        'horse': {
            'enc': (WEIGHT_HORSE_ENC_PATH, MODEL_HORSE_ENC_PATH),
            'dec': (WEIGHT_HORSE_DEC_PATH, MODEL_HORSE_DEC_PATH)
        },
        'church': {
            'enc': (WEIGHT_CHURCH_ENC_PATH, MODEL_CHURCH_ENC_PATH),
            'dec': (WEIGHT_CHURCH_DEC_PATH, MODEL_CHURCH_DEC_PATH)
        },
    }
    info = dic_model[model_type]
    WEIGHT_ENC_PATH, MODEL_ENC_PATH = info['enc']
    WEIGHT_DEC_PATH, MODEL_DEC_PATH = info['dec']

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
