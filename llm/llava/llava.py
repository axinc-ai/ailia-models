import itertools
import sys
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# from sample_utils import decode_batch, mask_to_bboxes, draw_bbox

# ======================
# Parameters
# ======================

WEIGHT_PATH = "llava-v1.5-7b.onnx"
MODEL_PATH = "llava-v1.5-7b.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/llava/"

IMAGE_PATH = "view.jpg"
SAVE_IMAGE_PATH = "output.png"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("LLaVA", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================


def preprocess(img):
    return img


def post_processing(output):
    return None


def forward(net, input_ids, inputs_embeds, position_ids, past_key_values):
    # feedforward
    if not args.onnx:
        output = net.predict([input_ids, inputs_embeds, position_ids, past_key_values])
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "past_key_values.0.decoder.key": past_key_values[0],
                "past_key_values.0.decoder.value": past_key_values[1],
                "past_key_values.0.encoder.key": past_key_values[2],
                "past_key_values.0.encoder.value": past_key_values[3],
                "past_key_values.1.decoder.key": past_key_values[4],
                "past_key_values.1.decoder.value": past_key_values[5],
                "past_key_values.1.encoder.key": past_key_values[6],
                "past_key_values.1.encoder.value": past_key_values[7],
                "past_key_values.2.decoder.key": past_key_values[8],
                "past_key_values.2.decoder.value": past_key_values[9],
                "past_key_values.2.encoder.key": past_key_values[10],
                "past_key_values.2.encoder.value": past_key_values[11],
                "past_key_values.3.decoder.key": past_key_values[12],
                "past_key_values.3.decoder.value": past_key_values[13],
                "past_key_values.3.encoder.key": past_key_values[14],
                "past_key_values.3.encoder.value": past_key_values[15],
                "past_key_values.4.decoder.key": past_key_values[16],
                "past_key_values.4.decoder.value": past_key_values[17],
                "past_key_values.4.encoder.key": past_key_values[18],
                "past_key_values.4.encoder.value": past_key_values[19],
                "past_key_values.5.decoder.key": past_key_values[20],
                "past_key_values.5.decoder.value": past_key_values[21],
                "past_key_values.5.encoder.key": past_key_values[22],
                "past_key_values.5.encoder.value": past_key_values[23],
                "past_key_values.6.decoder.key": past_key_values[24],
                "past_key_values.6.decoder.value": past_key_values[25],
                "past_key_values.6.encoder.key": past_key_values[26],
                "past_key_values.6.encoder.value": past_key_values[27],
                "past_key_values.7.decoder.key": past_key_values[28],
                "past_key_values.7.decoder.value": past_key_values[29],
                "past_key_values.7.encoder.key": past_key_values[30],
                "past_key_values.7.encoder.value": past_key_values[31],
                "past_key_values.8.decoder.key": past_key_values[32],
                "past_key_values.8.decoder.value": past_key_values[33],
                "past_key_values.8.encoder.key": past_key_values[34],
                "past_key_values.8.encoder.value": past_key_values[35],
                "past_key_values.9.decoder.key": past_key_values[36],
                "past_key_values.9.decoder.value": past_key_values[27],
                "past_key_values.9.encoder.key": past_key_values[38],
                "past_key_values.9.encoder.value": past_key_values[39],
                "past_key_values.10.decoder.key": past_key_values[40],
                "past_key_values.10.decoder.value": past_key_values[41],
                "past_key_values.10.encoder.key": past_key_values[42],
                "past_key_values.10.encoder.value": past_key_values[43],
                "past_key_values.11.decoder.key": past_key_values[44],
                "past_key_values.11.decoder.value": past_key_values[45],
                "past_key_values.11.encoder.key": past_key_values[46],
                "past_key_values.11.encoder.value": past_key_values[47],
                "past_key_values.12.decoder.key": past_key_values[48],
                "past_key_values.12.decoder.value": past_key_values[49],
                "past_key_values.12.encoder.key": past_key_values[50],
                "past_key_values.12.encoder.value": past_key_values[51],
                "past_key_values.13.decoder.key": past_key_values[52],
                "past_key_values.13.decoder.value": past_key_values[53],
                "past_key_values.13.encoder.key": past_key_values[54],
                "past_key_values.13.encoder.value": past_key_values[55],
                "past_key_values.14.decoder.key": past_key_values[56],
                "past_key_values.14.decoder.value": past_key_values[57],
                "past_key_values.14.encoder.key": past_key_values[58],
                "past_key_values.14.encoder.value": past_key_values[59],
                "past_key_values.15.decoder.key": past_key_values[60],
                "past_key_values.15.decoder.value": past_key_values[61],
                "past_key_values.15.encoder.key": past_key_values[62],
                "past_key_values.15.encoder.value": past_key_values[63],
            },
        )
    logits = output[0]
    past_key_values = output[1:]

    return logits, past_key_values


def prepare_inputs_labels_for_multimodal(
    models,
    input_ids,
    images,
):
    net = models["encode_images"]
    if not args.onnx:
        output = net.predict([images])
    else:
        output = net.run(
            None,
            {"images": images},
        )
    image_features = output[0]

    # Let's just add dummy tensors
    attention_mask = np.ones_like(input_ids, dtype=bool)
    labels = np.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    net = models["embed_tokens"]

    new_input_embeds = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]

            if not args.onnx:
                output = net.predict([cur_input_ids])
            else:
                output = net.run(
                    None,
                    {"input_ids": cur_input_ids},
                )
            cur_input_embeds_1 = output[0]

            cur_input_embeds = np.concatenate(
                [cur_input_embeds_1, cur_image_features[0:0]], axis=0
            )
            new_input_embeds.append(cur_input_embeds)
            cur_image_idx += 1
            continue

        image_token_indices = (
            [-1]
            + np.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
        split_sizes = [x.shape[0] for x in cur_labels_noim]

        input = np.concatenate(cur_input_ids_noim)
        if not args.onnx:
            output = net.predict([input])
        else:
            output = net.run(
                None,
                {"input_ids": input},
            )
        cur_input_embeds = output[0]

        cur_input_embeds_no_im = np.split(
            cur_input_embeds, list(itertools.accumulate(split_sizes)), axis=0
        )
        cur_new_input_embeds = []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)

        cur_new_input_embeds = [x for x in cur_new_input_embeds]
        cur_new_input_embeds = np.concatenate(cur_new_input_embeds)
        new_input_embeds.append(cur_new_input_embeds)

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)

    new_input_embeds_padded = []
    for i, cur_new_embed in enumerate(new_input_embeds):
        cur_len = cur_new_embed.shape[0]
        new_input_embeds_padded.append(
            np.concatenate(
                (
                    cur_new_embed,
                    np.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                    ),
                ),
                axis=0,
            )
        )

    new_input_embeds = np.stack(new_input_embeds_padded, axis=0)

    return new_input_embeds


def stopping_criteria(input_ids: np.array) -> bool:
    max_length = 513

    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    return is_done


def greedy_search(net, inputs_embeds, attention_mask):
    bos_token_id = 1
    pad_token_id = 0
    eos_token_id = np.array([2])

    shape = inputs_embeds.shape[:2]
    batch_size = shape[0]

    # _prepare_model_inputs
    input_ids = np.ones((batch_size, 1), dtype=int) * bos_token_id
    past_key_values = [
        np.zeros((1, 32, 0, 128), dtype=np.float16),
    ] * (32 * 2)

    # position_ids = np.load("position_ids.npy")
    # for i in range(32):
    #     for j in range(2):
    #         path = f"past_key_values_{i}_{j}.npy"
    #         past_key_values[i][j] = np.load(path)

    # keep track of which sequences are already finished
    unfinished_sequences = np.ones(input_ids.shape[0], dtype=int)

    this_peer_finished = False  # used by synced_gpus only
    while True:
        # prepare model inputs
        position_ids = np.cumsum(attention_mask, axis=-1) - 1
        position_ids[attention_mask == 0] = 1
        if 0 < past_key_values[0].shape[2]:
            position_ids = position_ids[:, -input_ids[:, 1:].shape[1] :]

        logits, past_key_values = forward(
            net,
            input_ids[:, 1:][:, -1:],
            inputs_embeds,
            position_ids,
            past_key_values,
        )

        next_tokens_scores = logits[:, -1, :]

        # argmax
        next_tokens = np.argmax(next_tokens_scores, axis=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)
        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1), dtype=int)], axis=-1
        )
        inputs_embeds = inputs_embeds[:, :0, :]

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences * np.prod(
            np.tile(next_tokens, (eos_token_id.shape[0], 1)) != eos_token_id[:, None],
            axis=0,
        )

        # stop when each sentence is finished
        if np.max(unfinished_sequences) == 0:
            this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids):
            this_peer_finished = True

        if this_peer_finished:
            break

    return input_ids


def predict(models, img):
    # img = preprocess(img)

    input_ids = np.load("input_ids.npy")
    images = np.load("images.npy")
    image_sizes = [(1000, 667)]
    inputs_embeds = prepare_inputs_labels_for_multimodal(
        models,
        input_ids,
        images,
    )
    attention_mask = np.ones(inputs_embeds.shape[:2], dtype=int)

    net = models["net"]
    generated_ids = greedy_search(net, inputs_embeds, attention_mask)

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
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                out = predict(models, img)
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
            out = predict(models, img)

        # res_img = draw_bbox(out)

        # # plot result
        # savepath = get_savepath(args.savepath, image_path, ext=".png")
        # logger.info(f"saved at : {savepath}")
        # cv2.imwrite(savepath, res_img)

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

        # cuda = 0 < ailia.get_gpu_environment_id()
        # providers = (
        # ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # if cuda
        # else ["CPUExecutionProvider"]
        # )
        # net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
        # net = onnxruntime.InferenceSession("onnx/xxx.onnx")
        encode_images = onnxruntime.InferenceSession("encode_images.onnx")
        embed_tokens = onnxruntime.InferenceSession("embed_tokens.onnx")

    models = {
        "net": net,
        "encode_images": encode_images,
        "embed_tokens": embed_tokens,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()