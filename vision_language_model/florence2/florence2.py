import sys
import time
from typing import List

# logger
from logging import getLogger  # noqa

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_EMB_PATH = "embeddings.onnx"
WEIGHT_ENC_IMG_PATH = "encode_image.onnx"
WEIGHT_ENC_BASE_PATH = "encoder_base.onnx"
WEIGHT_DEC_BASE_PATH = "decoder_base.onnx"
MODEL_EMB_PATH = "embeddings.onnx.prototxt"
MODEL_ENC_IMG_PATH = "encode_image.onnx.prototxt"
MODEL_ENC_BASE_PATH = "encoder_base.onnx.prototxt"
MODEL_DEC_BASE_PATH = "decoder_base.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/florence2/"

IMAGE_PATH = "car.jpg"
SAVE_IMAGE_PATH = "output.png"

IMG_SIZE = 768

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Florence-2", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Main functions
# ======================


def preprocess(img):
    h = w = IMG_SIZE

    img = np.array(Image.fromarray(img).resize((w, h), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float16)

    return img


def construct_prompts(text):
    task_prompts_without_inputs = {
        "<OCR>": "What is the text in the image?",
        "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
        "<CAPTION>": "What does the image describe?",
        "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
        "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
        "<OD>": "Locate the objects with category name in the image.",
        "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
        "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
    }
    task_prompts_with_input = {
        "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
        "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
        "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
        "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
        "<REGION_TO_CATEGORY>": "What is the region {input}?",
        "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
        "<REGION_TO_OCR>": "What text is in the region {input}?",
    }

    # replace the task tokens with the task prompts if task token is in the text
    prompts = []
    for _text in text:
        # 1. fixed task prompts without additional inputs
        for task_token, task_prompt in task_prompts_without_inputs.items():
            if task_token in _text:
                assert (
                    _text == task_token
                ), f"Task token {task_token} should be the only token in the text."
                _text = task_prompt
                break
        # 2. task prompts with additional inputs
        for task_token, task_prompt in task_prompts_with_input.items():
            if task_token in _text:
                _text = task_prompt.format(input=_text.replace(task_token, ""))
                break
        prompts.append(_text)

    return prompts


def predict(models, img, task_prompt, text_input=None):
    img = img[:, :, ::-1]  # BGR -> RGB
    pixel_values = preprocess(img)

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    text = construct_prompts([prompt])

    tokenizer = models["tokenizer"]
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=False,
        return_token_type_ids=False,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Extra the input embeddings
    net = models["embedding"]
    if not args.onnx:
        output = net.predict([input_ids])
    else:
        output = net.run(None, {"input_ids": input_ids})
    inputs_embeds = output[0]

    # Merge text and images
    net = models["encode_image"]
    if not args.onnx:
        output = net.predict([pixel_values])
    else:
        output = net.run(None, {"pixel_values": pixel_values})
    image_features = output[0]
    inputs_embeds = np.concatenate([image_features, inputs_embeds], axis=1)

    attention_mask = np.ones(inputs_embeds.shape[:2], dtype=int)

    net = models["encoder"]
    if not args.onnx:
        output = net.predict([inputs_embeds, attention_mask])
    else:
        output = net.run(
            None, {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        )
    last_hidden_state = output[0]


def recognize_from_image(models):
    prompt = "<OD>"

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
                output = predict(models, img, prompt)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Logging
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            output = predict(models, img, prompt)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_IMG_PATH, MODEL_ENC_IMG_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_BASE_PATH, MODEL_ENC_BASE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_BASE_PATH, MODEL_DEC_BASE_PATH, REMOTE_PATH)

    # seed = args.seed
    # if seed is not None:
    #     np.random.seed(seed)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        embedding = ailia.Net(
            MODEL_EMB_PATH,
            WEIGHT_EMB_PATH,
            env_id=env_id,
        )
        encode_image = ailia.Net(
            MODEL_ENC_IMG_PATH,
            WEIGHT_ENC_IMG_PATH,
            env_id=env_id,
        )
        encoder = ailia.Net(
            MODEL_ENC_BASE_PATH,
            WEIGHT_ENC_BASE_PATH,
            env_id=env_id,
        )
        decoder = ailia.Net(
            MODEL_DEC_BASE_PATH,
            WEIGHT_DEC_BASE_PATH,
            env_id=env_id,
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        embedding = onnxruntime.InferenceSession(WEIGHT_EMB_PATH, providers=providers)
        encode_image = onnxruntime.InferenceSession(
            WEIGHT_ENC_IMG_PATH, providers=providers
        )
        encoder = onnxruntime.InferenceSession(
            WEIGHT_ENC_BASE_PATH, providers=providers
        )
        decoder = onnxruntime.InferenceSession(
            WEIGHT_DEC_BASE_PATH, providers=providers
        )

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.BartTokenizerFast.from_pretrained("./tokenizer")
    else:
        raise NotImplementedError("ailia tokenizer is not supported yet.")

    models = {
        "tokenizer": tokenizer,
        "embedding": embedding,
        "encode_image": encode_image,
        "encoder": encoder,
        "decoder": decoder,
    }

    # generate
    recognize_from_image(models)


if __name__ == "__main__":
    main()
