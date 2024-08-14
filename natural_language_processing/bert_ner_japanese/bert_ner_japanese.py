import sys
import time
from logging import getLogger

import numpy as np

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "model.onnx"
MODEL_PATH = "model.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert_ner_japanese/"

TEXT_PATH = "sentences.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("bert-ner-japanese ", TEXT_PATH, None)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


# ======================
# Main functions
# ======================


def preprocess(tokenizer, sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="np",
        truncation=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=False,
    )

    num_chunks = len(inputs["input_ids"])
    for i in range(num_chunks):
        model_inputs = {k: np.expand_dims(v[i], axis=0) for k, v in inputs.items()}
        # model_inputs["sentence"] = sentence if i == 0 else None
        # model_inputs["is_last"] = i == num_chunks - 1

        yield model_inputs


def post_processing(all_outputs):
    pass


def predict(models, sentence):
    tokenizer = models["tokenizer"]
    net = models["net"]

    processed = preprocess(tokenizer, sentence)

    accumulator = []
    for item in processed:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        token_type_ids = item["token_type_ids"]

        # feedforward
        if not args.onnx:
            output = net.predict([input_ids, attention_mask, token_type_ids])
        else:
            output = net.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
        logits = output[0]

        processed["logits"] = logits
        accumulator.append(item)

    post_processing(accumulator)

    return


def recognize_from_sentence(models):
    sentence = "株式会社Jurabiは、東京都台東区に本社を置くIT企業である。"

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            processed = predict(models, sentence)
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
        processed = predict(models, sentence)

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

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    else:
        from ailia_tokenizer import Tokenizer

        # tokenizer = Tokenizer.from_pretrained("./tokenizer/")

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    recognize_from_sentence(models)


if __name__ == "__main__":
    main()
