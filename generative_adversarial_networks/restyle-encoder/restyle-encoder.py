import sys
import time

import numpy as np
import cv2
import onnx
import onnxruntime

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'restyle-encoder.onnx'
MODEL_PATH = 'restyle-encoder.onnx.prototxt'
#REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/restyle-encoder/'

IMAGE_PATH = 'face_img.jpg'
AVERAGE_IMAGE_PATH = 'avg_image.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Restyle Encoder', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-iter', '--iteration',
    default=5, type=int,
    help='Number of iterations per batch (default 5)'
)
parser.add_argument(
    '-d', '--debug',
    action='store_true',
    help='Debugger'
)
parser.add_argument(
    '--onnx_runtime',
    action='store_true',
    help='Inference using onnx runtime'
)
args = update_parser(parser)


# ======================
# Utils
# ======================
def check(onnx_model):
    onnx.checker.check_model(onnx_model)

def run_on_batch(inputs, net, iters, avg_image):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(iters):
        if iter == 0:
            #logger.info(avg_image.shape)
            avg_image_for_batch = np.tile(avg_image, (inputs.shape[0], 1, 1, 1))
            #logger.info(avg_image_for_batch.shape)
            x_input = np.concatenate((inputs, avg_image_for_batch), axis=1)
            #logger.info(x_input.shape)
        else:
            x_input = np.concatenate((inputs, y_hat), axis=1)
            #logger.info(x_input.shape)

        logger.info(f"Iteration {iter+1}/{iters}")
        y_hat, latent = net.forward(x_input, latent=latent)

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            #results_latent[idx].append(latent[idx].cpu().numpy())
            results_latent[idx].append(latent[idx])

        y_hat = net.face_pool(y_hat)

    return results_batch, results_latent

def np2im(var, input=False):
    var = var.astype('float32')
    var = cv2.cvtColor(
        var.transpose(1, 2, 0),
        cv2.COLOR_RGB2BGR
    )
    if not input:
        var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')

def recognize_from_image_onnx(filename, ort_session):

    #logger.info(filename)
    input_img = load_image(
        filename,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    #logger.info(input_img.shape)

    avg_img = load_image(
        AVERAGE_IMAGE_PATH,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    #logger.info(avg_img.shape)

    avg_img = (avg_img * 2) - 1

    avg_image_for_batch = np.tile(avg_img, (input_img.shape[0], 1, 1, 1))
    x_input = np.concatenate((input_img, avg_image_for_batch), axis=1)

    #logger.info(x_input.shape)

    ort_inputs = {ort_session.get_inputs()[0].name: x_input, 
                    ort_session.get_inputs()[1].name: np.random.randn(1, 18, 512).astype(np.float32)
                }

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            ort_outs = ort_session.run(None, ort_inputs)
            pred = ort_outs[0]
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        ort_outs = ort_session.run(None, ort_inputs)
        pred = ort_outs[0]

    logger.info(f'output shape: {pred.shape}')

def post_processing(result_batch, input_img):
    for i in range(input_img.shape[0]):
        results = [np2im(result_batch[i][iter_idx]) for iter_idx in range(args.iteration)]

        # save step-by-step results side-by-side
        input_im = np2im(input_img[i], input=True)
        #res = np.array(results[0].resize(resize_amount))
        res = np.array(results[0])
        for idx, result in enumerate(results[1:]):
            res = np.concatenate([res, result], axis=1)
        res = np.concatenate([res, input_im], axis=1)
    
    return res

# ======================
# Main functions
# ======================
def recognize_from_image(filename, net):

    #logger.info(filename)
    input_img = load_image(
        filename,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    #logger.info(input_img.shape)

    input_img_resized = load_image(
        filename,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    #logger.info(input_img_resized.shape)

    avg_img = load_image(
        AVERAGE_IMAGE_PATH,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True,
    )
    #logger.info(avg_img.shape)

    avg_img = (avg_img * 2) - 1
    
    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            result_batch, result_latents = run_on_batch(input_img_resized, net, args.iteration, avg_img)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        result_batch, result_latents = run_on_batch(input_img_resized, net, args.iteration, avg_img)

    res = post_processing(result_batch, input_img)
            
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, res)

    """
    # To be removed
    for i in range(input_img.shape[0]):
        input_im = np2im(input_img[i], input=True)
    savepath = get_savepath(args.savepath, filename)
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, input_im)
    """

def recognize_from_video(filename, net):

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    # Average image
    avg_img = load_image(
        AVERAGE_IMAGE_PATH,
        (RESIZE_HEIGHT, RESIZE_WIDTH),
        normalize_type='255',
        gen_input_ailia=True
    )
    avg_img = (avg_img * 2) - 1

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # Resize by padding the perimeter.
        _, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255'
        )

        resized_input = cv2.resize(input_data[0].transpose(1,2,0), (RESIZE_HEIGHT, RESIZE_WIDTH))
        resized_input = np.expand_dims(resized_input.transpose(2,0,1), axis=0)

        # inference
        result_batch, result_latents = run_on_batch(resized_input, net, args.iteration, avg_img)

        # post-processing
        res_img = post_processing(result_batch, input_data)

        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # debug
    if args.debug:
        onnx_model = onnx.load(WEIGHT_PATH)
        check(onnx_model)
        logger.info('Debug OK.')
    else:
        if args.video is not None:
            # net initialize
            net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
            # video mode
            recognize_from_video(SAVE_IMAGE_PATH, net)
        else:
            # image mode
            if args.onnx_runtime:
                # onnx runtime
                ort_session = onnxruntime.InferenceSession(WEIGHT_PATH)
                # input image loop
                for image_path in args.input:
                    recognize_from_image_onnx(image_path, ort_session)
            else:
                # net initialize
                net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
                #net = None
                # input image loop
                for image_path in args.input:
                    recognize_from_image(image_path, net)
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
