import sys
import time
from typing import List
import random

# logger
from logging import getLogger  # noqa

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from math_utils import softmax

from logit_process import logits_processor

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = "Qwen2-VL-2B.onnx"
WEIGHT_VIS_PATH = "Qwen2-VL-2B_vis.onnx"
MODEL_PATH = "Qwen2-VL-2B.onnx.prototxt"
MODEL_VIS_PATH = "Qwen2-VL-2B_vis.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen2_vl/"

IMAGE_PATH = "demo.jpg"
SAVE_IMAGE_PATH = "output.png"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Qwen2-VL", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="Describe this image.",
    help="prompt",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


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


def forward(
    net,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    past_key_values: List[np.ndarray],
):
    if not args.onnx:
        output = net.predict(
            [input_ids, inputs_embeds, position_ids, attention_mask, *past_key_values]
        )
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
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
                "key_cache24": past_key_values[48],
                "value_cache24": past_key_values[49],
                "key_cache25": past_key_values[50],
                "value_cache25": past_key_values[51],
                "key_cache26": past_key_values[52],
                "value_cache26": past_key_values[53],
                "key_cache27": past_key_values[54],
                "value_cache27": past_key_values[55],
            },
        )

    logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask):
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        position_ids = np.ones(
            (3, input_ids.shape[0], input_ids.shape[1]),
            dtype=input_ids.dtype,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = np.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = np.sum(vision_tokens == image_token_id)
            video_nums = np.sum(vision_tokens == video_token_id)
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
                )

                t_index = np.tile(
                    np.arange(llm_grid_t).reshape(-1, 1), (1, llm_grid_h * llm_grid_w)
                ).flatten()
                h_index = np.tile(
                    np.arange(llm_grid_h).reshape(1, -1, 1),
                    (llm_grid_t, 1, llm_grid_w),
                ).flatten()
                w_index = np.tile(
                    np.arange(llm_grid_w).reshape(1, 1, -1),
                    (llm_grid_t, llm_grid_h, 1),
                ).flatten()
                llm_pos_ids_list.append(
                    np.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
                )

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )

        mrope_position_deltas = np.expand_dims(np.array(mrope_position_deltas), axis=1)
        return position_ids, mrope_position_deltas
    else:
        raise NotImplementedError


def stopping_criteria(input_ids: np.array) -> np.array:
    max_length = 438

    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = np.array([151645, 151643])
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def sample(models):
    pad_token_id = 151643
    image_token_id = 151655
    image_token_id = np.array([image_token_id])
    video_grid_thw = None

    net = models["visual"]
    if not args.onnx:
        output = net.predict(
            [input_ids, pixel_values, image_grid_thw, attention_mask, image_token_id]
        )
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "image_token_id": image_token_id,
            },
        )
    inputs_embeds = output[0]
    past_key_values = [
        np.zeros((1, 2, 0, 128), dtype=np.float32) for _ in range(28 * 2)
    ]

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = np.ones(batch_size, dtype=int)

    net = models["net"]
    while True:
        # prepare model inputs
        model_input_ids = input_ids
        if model_input_ids.shape[1] != cache_position.shape[0]:
            model_input_ids = model_input_ids[:, cache_position]
        batch_size, seq_length = model_input_ids.shape
        if cache_position[0] == 0:
            position_ids, rope_deltas = get_rope_index(
                model_input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
        else:
            delta = cache_position[0] + rope_deltas if rope_deltas is not None else 0
            position_ids = np.arange(seq_length)
            position_ids = np.tile(position_ids.reshape(1, -1), (batch_size, 1))
            position_ids = position_ids + delta
            position_ids = np.tile(np.expand_dims(position_ids, axis=0), (3, 1, 1))
        if cache_position[0] != 0:
            # Disable inputs_embeds parameter if cache is used
            inputs_embeds = inputs_embeds[:0, :1, :]

        logits, past_key_values = forward(
            net,
            model_input_ids,
            inputs_embeds,
            position_ids,
            attention_mask,
            past_key_values,
        )

        next_token_logits = logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # token selection
        probs = softmax(next_token_scores, axis=-1)
        next_tokens = np.random.choice(len(probs[0]), size=1, p=probs[0])

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)

        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=int)],
            axis=-1,
        )
        cache_position = cache_position[-1:] + 1

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids)
        this_peer_finished = np.max(unfinished_sequences) == 0
        cur_len += 1

        if this_peer_finished:
            break


def predict(models, img, prompt):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB
    # pixel_values = preprocess(img)

    # tokenizer = models["tokenizer"]
    # inputs = tokenizer(
    #     text,
    #     return_tensors="np",
    #     padding=False,
    #     return_token_type_ids=False,
    # )
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    input_ids = np.load("input_ids.npy")

    # # Extra the input embeddings
    # net = models["embedding"]
    # if not args.onnx:
    #     output = net.predict([input_ids])
    # else:
    #     output = net.run(None, {"input_ids": input_ids})
    # inputs_embeds = output[0]

    # # Merge text and images
    # net = models["encode_image"]
    # if not args.onnx:
    #     output = net.predict([pixel_values])
    # else:
    #     output = net.run(None, {"pixel_values": pixel_values})
    # image_features = output[0]
    # inputs_embeds = np.concatenate([image_features, inputs_embeds], axis=1)

    # attention_mask = np.ones(inputs_embeds.shape[:2], dtype=int)

    result = sample(models)

    # tokenizer = models["tokenizer"]
    # generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # answer = post_process_generation(
    #     generated_text, task=prompt, image_size=(im_w, im_h)
    # )
    # return answer


def recognize_from_image(models):
    prompt = args.prompt
    logger.info("Prompt: %s" % prompt)

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
                output_text = predict(models, img, prompt)
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
            output_text = predict(models, img, prompt)

        print(output_text)

    logger.info("Script finished successfully.")


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VIS_PATH, MODEL_VIS_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        visual = ailia.Net(
            "Qwen2-VL-2B_vis.onnx.prototxt", "Qwen2-VL-2B_vis.onnx", env_id=env_id
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        visual = onnxruntime.InferenceSession(WEIGHT_VIS_PATH, providers=providers)
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    models = {
        "visual": visual,
        "net": net,
    }

    # generate
    recognize_from_image(models)


if __name__ == "__main__":
    main()
