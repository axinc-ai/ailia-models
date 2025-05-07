import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models
from image_utils import normalize_image
from detector_utils import load_image
from math_utils import softmax

# logger
from logging import getLogger


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "regnet_y_800mf.onnx"
MODEL_PATH = "regnet_y_800mf.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/siglip/"

IMAGE_PATH = "test.jpg"
SAVE_IMAGE_PATH = "output.png"

LABELS_FILE_PATH = "imagenet_2012.txt"

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SigLIP2", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RBG

    # resize
    short, long = (im_w, im_h) if im_w <= im_h else (im_h, im_w)
    requested_new_short = 232
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    ow, oh = (new_short, new_long) if im_w <= im_h else (new_long, new_short)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BILINEAR))

    # center_crop
    if ow > w:
        x = int(round((ow - w) / 2.0))
        img = img[:, x : x + w, :]
    if oh > h:
        y = int(round((oh - h) / 2.0))
        img = img[y : y + h, :, :]

    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def postprocess_result(output: np.ndarray, top_k: int = 5):
    softmaxed_scores = softmax(output, -1)[0]
    topk_labels = np.argsort(softmaxed_scores)[-top_k:][::-1]
    topk_scores = softmaxed_scores[topk_labels]

    return topk_labels, topk_scores


def predict(net, img):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {"input": img})
    result = output[0]

    return result


def recognize_from_image(net):
    with open(LABELS_FILE_PATH) as f:
        imagenet_classes = [x.strip() for x in f.readlines() if x.strip()]

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = predict(net, img)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Loggin
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            result = predict(net, img)

        top_labels, top_scores = postprocess_result(result)

        # Show results
        for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
            _, predicted_label = imagenet_classes[label].split(" ", 1)
            print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_image(net)


if __name__ == "__main__":
    main()
