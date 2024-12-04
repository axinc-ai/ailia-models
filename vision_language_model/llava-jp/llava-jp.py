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
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa
from detector_utils import load_image  # noqa
from math_utils import softmax

from logit_process import logits_processor

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/llava-jp/"

IMAGE_PATH = "sample.jpg"
SAVE_IMAGE_PATH = "output.png"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("LLaVA-JP", IMAGE_PATH, SAVE_IMAGE_PATH, large_model=True)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="猫の隣には何がありますか？",
    help="prompt",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Model selection
# ======================


WEIGHT_PATH = "llava-jp-1.3b-v1.1.onnx"
MODEL_PATH = "llava-jp-1.3b-v1.1.onnx.prototxt"


# ======================
# Secondary Functions
# ======================


# ======================
# Main functions
# ======================


def prepare_inputs_for_multimodal(
    input_ids, position_ids, attention_mask, past_key_values, images
):
    if input_ids.shape[1] == 1:
        target_shape = past_key_values[-1][-1].shape[-2] + 1
        attention_mask = np.concatenate(
            (
                attention_mask,
                np.ones(
                    (
                        attention_mask.shape[0],
                        target_shape - attention_mask.shape[1],
                    ),
                    dtype=attention_mask.dtype,
                ),
            ),
            axis=1,
        )
        position_ids = np.expand_dims(np.sum(attention_mask, axis=1), axis=-1) - 1
        return (
            input_ids,
            position_ids,
            past_key_values,
            None,
        )

    # image_features = encode_images(images)

    new_input_embeds = np.load("new_input_embeds.npy")
    max_len = new_input_embeds[0].shape[0]

    position_ids = np.zeros(
        (1, max_len),
        dtype=position_ids.dtype,
    )
    cur_len = 781
    if cur_len > 0:
        attention_mask[0, :cur_len] = True
        position_ids[0, :cur_len] = np.arange(
            0,
            cur_len,
            dtype=position_ids.dtype,
        )

    return (
        None,
        position_ids,
        past_key_values,
        new_input_embeds,
    )


def forward(
    net,
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    images: np.ndarray,
    past_key_values: List[np.ndarray],
    first_run,
):
    (
        input_ids,
        position_ids,
        past_key_values,
        inputs_embeds,
    ) = prepare_inputs_for_multimodal(
        input_ids, position_ids, attention_mask, past_key_values, images
    )

    if input_ids is None:
        input_ids = np.zeros((1, 0), dtype=np.int64)
    if inputs_embeds is None:
        inputs_embeds = np.zeros((1, 0, 2048), dtype=np.float32)

    if not args.onnx:
        # if first_run or COPY_BLOB_DATA == False:
        if 1:
            output = net.predict(
                [
                    input_ids,
                    position_ids,
                    inputs_embeds,
                    *past_key_values,
                ]
            )
            logits, new_past_key_values = output[0], output[1:]
        else:
            NUM_KV = 28
            key_shapes = []
            value_shapes = []
            for i in range(NUM_KV):
                key_shapes.append(
                    net.get_blob_shape(
                        net.find_blob_index_by_name("key_cache_out" + str(i))
                    )
                )
                value_shapes.append(
                    net.get_blob_shape(
                        net.find_blob_index_by_name("value_cache_out" + str(i))
                    )
                )
            net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data(
                inputs_embeds, net.find_blob_index_by_name("inputs_embeds")
            )
            net.set_input_blob_data(
                position_ids, net.find_blob_index_by_name("position_ids")
            )
            net.set_input_blob_data(
                attention_mask, net.find_blob_index_by_name("attention_mask")
            )
            for i in range(NUM_KV):
                net.set_input_blob_shape(
                    key_shapes[i], net.find_blob_index_by_name("key_cache" + str(i))
                )
                net.set_input_blob_shape(
                    value_shapes[i], net.find_blob_index_by_name("value_cache" + str(i))
                )
                net.copy_blob_data("key_cache" + str(i), "key_cache_out" + str(i))
                net.copy_blob_data("value_cache" + str(i), "value_cache_out" + str(i))
            net.update()
            logits = net.get_blob_data(net.find_blob_index_by_name("logits"))
            new_past_key_values = None
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "inputs_embeds": inputs_embeds,
                "key_cache0": past_key_values[0],
                "value_cache0": past_key_values[1],
                "key_cache1": past_key_values[2],
                "value_cache1": past_key_values[3],
                "key_cache2": past_key_values[4],
                "value_cache2": past_key_values[5],
                "key_cache3": past_key_values[6],
                "value_cache3": past_key_values[7],
                "key_cache4": past_key_values[8],
                "value_cache4": past_key_values[9],
                "key_cache5": past_key_values[10],
                "value_cache5": past_key_values[11],
                "key_cache6": past_key_values[12],
                "value_cache6": past_key_values[13],
                "key_cache7": past_key_values[14],
                "value_cache7": past_key_values[15],
                "key_cache8": past_key_values[16],
                "value_cache8": past_key_values[17],
                "key_cache9": past_key_values[18],
                "value_cache9": past_key_values[19],
                "key_cache10": past_key_values[20],
                "value_cache10": past_key_values[21],
                "key_cache11": past_key_values[22],
                "value_cache11": past_key_values[23],
                "key_cache12": past_key_values[24],
                "value_cache12": past_key_values[25],
                "key_cache13": past_key_values[26],
                "value_cache13": past_key_values[27],
                "key_cache14": past_key_values[28],
                "value_cache14": past_key_values[29],
                "key_cache15": past_key_values[30],
                "value_cache15": past_key_values[31],
                "key_cache16": past_key_values[32],
                "value_cache16": past_key_values[33],
                "key_cache17": past_key_values[34],
                "value_cache17": past_key_values[35],
                "key_cache18": past_key_values[36],
                "value_cache18": past_key_values[37],
                "key_cache19": past_key_values[38],
                "value_cache19": past_key_values[39],
                "key_cache20": past_key_values[40],
                "value_cache20": past_key_values[41],
                "key_cache21": past_key_values[42],
                "value_cache21": past_key_values[43],
                "key_cache22": past_key_values[44],
                "value_cache22": past_key_values[45],
                "key_cache23": past_key_values[46],
                "value_cache23": past_key_values[47],
            },
        )
        logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def stopping_criteria(input_ids: np.array, max_length) -> np.array:
    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = np.array([151645, 151643])
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def tokenizer_decode(input_ids, generated_ids, tokenizer, intermediate):
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    try:
        if args.disable_ailia_tokenizer:
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=False,
            )
    except UnicodeDecodeError:
        if intermediate:
            return [""]
        raise
    return output_text


def sample(
    models,
    input_ids,
    pixel_values,
    attention_mask,
    image_grid_thw,
    video_grid_thw,
    tokenizer,
):
    pad_token_id = 7

    if args.benchmark:
        start = int(round(time.time() * 1000))

    if INTERMEDIATE:
        print("Encoding..." + "\n\u001B[2A")
        before_text = ""

    net = models["visual"]
    if not args.onnx:
        output = net.predict([input_ids, pixel_values, image_grid_thw, image_token_id])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw.astype(np.int64),
                "image_token_id": image_token_id.astype(np.int64),
            },
        )
    inputs_embeds = output[0]
    past_key_values = [
        np.zeros((1, 2, 0, 128), dtype=np.float32) for _ in range(28 * 2)
    ]

    if args.benchmark:
        end = int(round(time.time() * 1000))
        estimation_time = end - start
        logger.info(f"\tencode time {estimation_time} ms")

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = np.ones(batch_size, dtype=int)
    cache_position = (
        np.cumsum(np.ones_like(input_ids[0, :], dtype=np.int64), axis=0) - 1
    )
    # max_length = args.max_length + input_ids.shape[1]

    net = models["net"]
    first_run = True
    while True:
        # prepare model inputs
        if 0 < past_key_values[0].shape[-2]:
            model_input_ids = input_ids[:, cache_position]
        else:
            model_input_ids = input_ids
        position_ids = attention_mask.astype(np.int32).cumsum(axis=-1) - 1
        if 0 < past_key_values[0].shape[-2]:
            position_ids = position_ids[:, -model_input_ids.shape[1] :]

        if args.benchmark:
            start = int(round(time.time() * 1000))

        logits, past_key_values = forward(
            net,
            model_input_ids,
            position_ids,
            attention_mask,
            images,
            past_key_values,
            first_run,
        )
        first_run = False

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tdecode time {estimation_time} ms")

        next_token_logits = logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(
            input_ids, next_token_logits, args.temperature, args.top_p, args.top_k
        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=int)],
            axis=-1,
        )
        cache_position = cache_position[-1:] + 1

        # token selection
        probs = softmax(next_token_scores, axis=-1)
        next_tokens = np.random.choice(len(probs[0]), size=1, p=probs[0])

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, max_length
        )
        this_peer_finished = np.max(unfinished_sequences) == 0
        cur_len += 1

        if this_peer_finished:
            break

        if INTERMEDIATE:
            output_text = tokenizer_decode(initial_ids, input_ids, tokenizer, True)[0]
            if output_text.startswith(before_text):
                deltaText = output_text[len(before_text) :]
            else:
                deltaText = output_text
            print(deltaText, end="")
            sys.stdout.flush()
            if output_text != "":
                before_text = output_text

    return input_ids


def predict(models, messages):
    generated_ids = sample(
        models,
        input_ids,
        pixel_values,
        attention_mask,
        image_grid_thw,
        video_grid_thw,
        tokenizer,
    )

    output_text = tokenizer_decode(input_ids, generated_ids, tokenizer, False)

    return output_text[0]


def recognize(models):
    prompt = args.prompt
    logger.info("Prompt: %s" % prompt)

    content = []
    if args.video is not None:
        content.append({"type": "video", "video": args.video})
    else:
        for input_path in args.input:
            content.append({"type": "image", "image": input_path})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # inference
    logger.info("Start inference...")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output_text = predict(models, messages)
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
        output_text = predict(models, messages)

    if INTERMEDIATE:
        print("")
    else:
        print(output_text)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VIS_PATH, MODEL_VIS_PATH, REMOTE_PATH)
    # if PB_PATH is not None:
    #     check_and_download_file(PB_PATH, REMOTE_PATH)
    # if PB_VIS_PATH is not None:
    #     check_and_download_file(PB_VIS_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        # visual = ailia.Net(
        #     MODEL_VIS_PATH, WEIGHT_VIS_PATH, env_id=env_id, memory_mode=memory_mode
        # )
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # visual = onnxruntime.InferenceSession(WEIGHT_VIS_PATH, providers=providers)
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    # # args.disable_ailia_tokenizer = True
    # if args.disable_ailia_tokenizer:
    #     import transformers

    #     tokenizer = transformers.Qwen2TokenizerFast.from_pretrained("./tokenizer")
    # else:
    #     from ailia_tokenizer import GPT2Tokenizer

    #     tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
    #     tokenizer.add_special_tokens(
    #         {
    #             "additional_special_tokens": [
    #                 "<|end_of_text|>",
    #                 "<|im_start|>",
    #                 "<|im_end|>",
    #                 "<|object_ref_start|>",
    #                 "<|object_ref_end|>",
    #                 "<|box_start|>",
    #                 "<|box_end|>",
    #                 "<|quad_start|>",
    #                 "<|quad_end|>",
    #                 "<|vision_start|>",
    #                 "<|vision_end|>",
    #                 "<|vision_pad|>",
    #                 "<|image_pad|>",
    #                 "<|video_pad|>",
    #             ]
    #         }
    #     )

    models = {
        # "tokenizer": tokenizer,
        # "visual": visual,
        "net": net,
    }

    # generate
    recognize(models)


if __name__ == "__main__":
    main()
