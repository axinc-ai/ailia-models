import sys
import time

import numpy as np
import cv2
from skimage.transform import resize

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

WEIGHT_PATH = 'group_vit_gcc_yfcc_30e-74d335e6.onnx'
MODEL_PATH = 'group_vit_gcc_yfcc_30e-74d335e6.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/group_vit/'

IMAGE_PATH = 'voc.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GroupViT', IMAGE_PATH, SAVE_IMAGE_PATH
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

def generate_composite_img(img, alpha_channel):
    b_channel, g_channel, r_channel = cv2.split(img)
    b_channel = b_channel * alpha_channel
    g_channel = g_channel * alpha_channel
    r_channel = r_channel * alpha_channel
    alpha_channel = (alpha_channel * 255).astype(b_channel.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return img_BGRA


# ======================
# Main functions
# ======================


def preprocess(img):
    img = normalize_image(img, normalize_type='ImageNet')

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def post_processing(output):
    pred_global, pred_local, pred_fusion = output

    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local[0, 0, :, :]
    pred_fusion = pred_fusion[0, 0, :, :]

    return pred_global, pred_local, pred_fusion


def predict(net, img):
    h, w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)
    print(img)
    print(img.shape, img.dtype)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'img': img})
    seg_logit = output[0]

    print(seg_logit)
    print(seg_logit.shape)

    # pred_glance, pred_focus, pred_fusion = post_processing(output)

    # return pred_glance, pred_focus, pred_fusion


def recognize_from_image(net):
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
                out = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            out = predict(net, img)

        # res_img = generate_composite_img(img, pred)

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


def recognize_from_video(net):
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
        out = predict(net, img)
        pred = out[2]

        # plot result
        res_img = generate_composite_img(img, pred)

        # show
        cv2.imshow('frame', res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
