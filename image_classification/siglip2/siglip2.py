import sys
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image
from transformers import AutoTokenizer

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models
from detector_utils import load_image
from math_utils import softmax


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_BASE_P16_224_PATH = "siglip2-base-patch16-224.onnx"
MODEL_BASE_P16_224_PATH = "siglip2-base-patch16-224.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/siglip/"

IMAGE_PATH = "demo.jpg"

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SigLIP2", IMAGE_PATH, None)
parser.add_argument(
    "-t",
    "--text",
    dest="text_inputs",
    type=str,
    action="append",
    help="Input text. (can be specified multiple times)",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="base-patch16-224",
    choices=("base-patch16-224", "large-patch16-256"),
    help="model type",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def preprocess(img):
    img = img[:, :, ::-1]  # BGR -> RBG

    # resize
    img = np.array(
        Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    )

    rescale_factor = 0.00392156862745098
    img = img.astype(np.float32) * rescale_factor
    img = (img - 0.5) / 0.5

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return img


def postprocess_result(logits_per_image, top_k: int = 5):
    probs = softmax(logits_per_image, axis=1)[0]

    top_labels = np.argsort(-probs)[: min(top_k, probs.shape[0])]
    top_probs = probs[top_labels]

    return top_labels, top_probs


def predict(net, img, input_ids):
    pixel_values = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, pixel_values])
    else:
        output = net.run(None, {"input_ids": input_ids, "pixel_values": pixel_values})
    logits_per_image = output[0]

    return logits_per_image


def recognize_from_image(models):
    input_labels = args.text_inputs
    if input_labels is None:
        input_labels = ["2 cats", "a plane", "a remote", "3 dogs"]

    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

    tokenizer = models["tokenizer"]
    encoded = tokenizer(
        text_descriptions,
        return_tensors="np",
        padding=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"]

    net = models["net"]

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
                logits_per_image = predict(net, img, input_ids)
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
            logits_per_image = predict(net, img, input_ids)

        top_labels, top_probs = postprocess_result(logits_per_image)

        # Show results
        a = [(input_labels[x], y) for x, y in zip(top_labels, top_probs)]
        for idx, (label, score) in enumerate(a):
            print(f"{idx + 1}: {label} - {score * 100 :.2f}%")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    dic_model = {
        "base-patch16-224": (WEIGHT_BASE_P16_224_PATH, MODEL_BASE_P16_224_PATH),
    }
    WEIGTH_PATH, MODEL_PATH = dic_model[args.model_type]

    check_and_download_models(WEIGTH_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGTH_PATH, env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGTH_PATH)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")

    models = dict(tokenizer=tokenizer, net=net)

    recognize_from_image(models)


if __name__ == "__main__":
    main()
