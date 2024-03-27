import sys
import os
import time
from logging import getLogger

import numpy as np
from transformers import AutoTokenizer
from scapy.all import Ether, CookedLinux

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "model.onnx"
MODEL_PATH = "model.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/bert-network-packet-flow-header-payload/"

PACEKT_HEX_PATH = "input_hex.txt"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "bert-network-packet-flow-header-payload",
    PACEKT_HEX_PATH,
    None,
)
parser.add_argument("--hex", type=str, default=None, help="Input-HEX data.")
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


# ======================
# Main functions
# ======================


def predict(models, packet_hex):
    packet_hex = "34 23 17914 -1 58804 25 322 282 62 0 5 3 -1 101 107 70 115 97 49 70 68 86 87 74 75 97 72 82 83 82 109 116 70 10 83 50 108 85 87 107 120 112 83 72 78 84 83 107 120 71 101 69 108 74 101 107 78 105 101 72 112 114 98 109 74 66 90 88 78 86 99 87 57 78 83 87 86 81 83 87 116 54 98 69 104 85 83 51 90 110 84 110 100 119 10 83 88 86 69 100 72 66 111 98 72 90 116 84 85 108 117 85 69 100 72 98 48 53 76 101 88 74 68 99 110 112 68 81 87 57 68 98 72 70 113 90 110 82 66 83 50 100 119 83 87 86 80 82 51 74 84 87 85 120 76 10 83 71 90 120 85 109 90 85 83 109 53 86 87 88 78 68 87 71 112 76 90 71 82 107 81 109 116 76 100 108 78 119 97 87 78 85 84 108 112 107 83 108 104 121 90 87 86 72 90 70 112 74 97 109 108 80 85 109 78 68 10 84 110 100 97 99 51 90 120 89 48 116 76 101 109 86 68 101 85 112 87 97 72 66 71 98 51 108 52 90 48 53 116 89 110 104 81 84 107 108 73 84 71 116 79 10 13 10 45 45 95 57 48 54 50 57 56 52 55 55 52 51 54 55 56 52 53 48 55 56 55 56 53 51 51 45 45 13 10 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"

    tokenizer = models["tokenizer"]
    model_inputs = tokenizer(packet_hex, return_tensors="np")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    net = models["net"]

    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, attention_mask])
    else:
        output = net.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
    logits = output[0]

    return (None,)


def recognize_from_packet(models):
    packet_hex = args.hex
    if packet_hex:
        args.input[0] = packet_hex

    # input audio loop
    for packet_path in args.input:
        # prepare input data
        if os.path.isfile(packet_path):
            logger.info(packet_path)
            with open(packet_path, "r") as f:
                packet_hex = f.read()

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, packet_hex)
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
            output = predict(models, packet_hex)

    tags = output[0]

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        pass
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")

    models = {
        "tokenizer": tokenizer,
        "net": net,
    }

    recognize_from_packet(models)


if __name__ == "__main__":
    main()
