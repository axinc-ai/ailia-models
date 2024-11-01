import sys
import time

# logger
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
from classifier_utils import plot_results, print_results  # noqa: E402
from math_utils import softmax  # noqa: E402C
import webcamera_utils  # noqa: E402


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_IMAGE_PATH = "encode_image.onnx"
MODEL_IMAGE_PATH = "encode_image.onnx.prototxt"
WEIGHT_TEXT_PATH = "encode_text.onnx"
MODEL_TEXT_PATH = "encode_text.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/clip-japanese-base/"

IMAGE_PATH = "chelsea.png"
SAVE_IMAGE_PATH = "output.png"

IMAGE_SIZE = 224

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("CLIP", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-t",
    "--text",
    dest="text_inputs",
    type=str,
    action="append",
    help="Input text. (can be specified multiple times)",
)
parser.add_argument(
    "--desc_file", default=None, metavar="DESC_FILE", type=str, help="description file"
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Main functions
# ======================


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = round(im_w * scale), round(im_h * scale)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x : x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y : y + h, :, :]

    img = img[:, :, ::-1]  # BGR -> RBG
    img = img / 255

    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img, text_features):
    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {"pixel_values": img})
    image_features = output[0]

    text_probs = 100.0 * image_features @ text_features.T

    pred = softmax(text_probs, axis=-1)

    return pred[0]


def get_text_features(models, text):
    tokenizer = models["tokenizer"]
    text_inputs = tokenizer(text)
    if not isinstance(text_inputs["input_ids"], np.ndarray):
        text_inputs = {k: v.numpy() for k, v in text_inputs.items()}

    net = models["text"]
    if not args.onnx:
        output = net.predict(
            [
                text_inputs["input_ids"],
                text_inputs["attention_mask"],
                text_inputs["position_ids"],
            ]
        )
    else:
        output = net.run(None, text_inputs)
    text_features = output[0]

    return text_features


def recognize_from_image(models):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = ["犬", "猫", "象"]

    text_features = get_text_features(models, text_inputs)

    net = models["image"]

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
                pred = predict(net, img, text_features)
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
            pred = predict(net, img, text_features)

        # show results
        pred = np.expand_dims(pred, axis=0)
        print_results(pred, text_inputs)

    logger.info("Script finished successfully.")


def recognize_from_video(models):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = ["犬", "猫", "象"]

    text_features = get_text_features(models, text_inputs)

    net = models["image"]

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        img = frame

        pred = predict(net, img, text_features)

        plot_results(frame, np.expand_dims(pred, axis=0), text_inputs)

        cv2.imshow("frame", frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_TEXT_PATH, MODEL_TEXT_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_image = ailia.Net(MODEL_IMAGE_PATH, WEIGHT_IMAGE_PATH, env_id=env_id)
        net_text = ailia.Net(MODEL_TEXT_PATH, WEIGHT_TEXT_PATH, env_id=env_id)
    else:
        import onnxruntime

        net_image = onnxruntime.InferenceSession(WEIGHT_IMAGE_PATH)
        net_text = onnxruntime.InferenceSession(WEIGHT_TEXT_PATH)

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("./tokenizer", trust_remote_code=True)
    else:
        raise NotImplementedError("ailia tokenizer is not supported.")

    models = {
        "tokenizer": tokenizer,
        "image": net_image,
        "text": net_text,
    }

    if args.video is not None:
        # video mode
        recognize_from_video(models)
    else:
        # image mode
        recognize_from_image(models)


if __name__ == "__main__":
    main()
