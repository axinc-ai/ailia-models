import sys
import time

import numpy as np
import cv2
from PIL import Image

from transformers import AutoTokenizer

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
# from image_utils import load_image, normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
# logger
from logging import getLogger  # noqa

from bert_model import language_backbone

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'xxx.onnx'
MODEL_PATH = 'xxx.onnx.prototxt'
WEIGHT_XXX_PATH = 'xxx.onnx'
MODEL_XXX_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/glip/'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'GLIP', IMAGE_PATH, SAVE_IMAGE_PATH
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

def preprocess(img, image_shape):
    pass


def post_processing(output):
    return None


def predict(models, img):
    # shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    # img = preprocess(img, shape)

    captions = ['bobble heads on top of the shelf']

    tokenizer = models["tokenizer"]
    max_length = 256

    tokenized = tokenizer.batch_encode_plus(
        captions,
        max_length=max_length,
        padding='max_length',
        return_special_tokens_mask=True,
        return_tensors='pt',
        truncation=True)

    net = models['language_backbone']
    language_dict_features = language_backbone(
        net,
        tokenized.input_ids.numpy(),
        tokenized.attention_mask.numpy(),
        tokenized.token_type_ids.numpy())

    print(language_dict_features["aggregate"])
    print(language_dict_features["aggregate"].shape)

    # # feedforward
    # if not args.onnx:
    #     output = net.predict([img])
    # else:
    #     output = net.run(None, {'src': img})
    #
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

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext='.png')
        # logger.info(f'saved at : {savepath}')
        # cv2.imwrite(savepath, res_img)

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
        out = predict(net, img)

        # plot result
        res_img = draw_bbox(frame, out)

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
    WEIGHT_PATH = "backbone.onnx"
    MODEL_PATH = "backbone.onnx.prototxt"
    WEIGHT_BERT_PATH = "language_backbone.onnx"
    MODEL_BERT_PATH = "language_backbone.onnx.prototxt"
    WEIGHT_RPN_PATH = "rpn.onnx"
    MODEL_RPN_PATH = "rpn.onnx.prototxt"

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_BERT_PATH, MODEL_BERT_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_RPN_PATH, MODEL_RPN_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        backbone = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        language_backbone = ailia.Net(MODEL_BERT_PATH, WEIGHT_BERT_PATH, env_id=env_id)
        rpn = ailia.Net(MODEL_RPN_PATH, WEIGHT_RPN_PATH, env_id=env_id)
    else:
        import onnxruntime
        backbone = onnxruntime.InferenceSession(WEIGHT_PATH)
        language_backbone = onnxruntime.InferenceSession(WEIGHT_BERT_PATH)
        rpn = onnxruntime.InferenceSession(WEIGHT_RPN_PATH)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    models = dict(
        tokenizer=tokenizer,
        backbone=backbone,
        language_backbone=language_backbone,
        rpn=rpn,
    )

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == '__main__':
    main()
