import sys
import time
import copy
import json
import re
from logging import getLogger
from typing import List

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from detector_utils import load_image
import webcamera_utils
from model_utils import check_and_download_models
from arg_utils import get_base_parser, update_parser, get_savepath

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/donut/"
WEIGHT_PATH = "donut-base-finetuned-cord-v2.onnx"
MODEL_PATH = "donut-base-finetuned-cord-v2.onnx.prototxt"

IMAGE_PATH = "cord_sample_receipt1.png"

COPY_BLOB_DATA = True


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Donut: Document Understanding Transformer", IMAGE_PATH, None)
parser.add_argument(
    "-w", "--write_results", action="store_true", help="Flag to output results to file."
)
args = update_parser(parser)


# ======================
# Utils
# ======================


def build_post_process(config):
    support_dict = ["DBPostProcess", "CTCLabelDecode", "ClsPostProcess"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "post process only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)

    return module_class


def token2json(tokenizer, tokens, is_inner_value=False):
    """
    Convert a (generated) token seuqnce into an ordered JSON format
    """
    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(tokenizer, content, is_inner_value=True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if (
                            leaf in tokenizer.get_added_vocab()
                            and leaf[0] == "<"
                            and leaf[-2:] == "/>"
                        ):
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokenizer, tokens[6:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


# ======================
# Main functions
# ======================


def forward(
    net,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    encoder_hidden_states: np.ndarray,
    past_key_values: List[np.ndarray],
    blob_copy: bool,
):
    if 1:  # not args.onnx:
        if not blob_copy:
            output = net.predict(
                [
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    *past_key_values,
                ]
            )
            logits, new_past_key_values = output[0], output[1:]
        else:
            NUM_KV = 8
            key_shapes = [
                net.get_blob_shape(
                    net.find_blob_index_by_name("key_cache_out" + str(i))
                )
                for i in range(NUM_KV)
            ]
            value_shapes = [
                net.get_blob_shape(
                    net.find_blob_index_by_name("value_cache_out" + str(i))
                )
                for i in range(NUM_KV)
            ]
            net.set_input_blob_data(input_ids, net.find_blob_index_by_name("input_ids"))
            net.set_input_blob_data(
                attention_mask, net.find_blob_index_by_name("attention_mask")
            )
            net.set_input_blob_data(
                encoder_hidden_states,
                net.find_blob_index_by_name("encoder_hidden_states"),
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
            new_past_key_values = [
                net.get_blob_data(net.find_blob_index_by_name("key_cache_out0"))
            ]
    else:
        output = net.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audios": audios,
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
                "key_cache28": past_key_values[56],
                "value_cache28": past_key_values[57],
                "key_cache29": past_key_values[58],
                "value_cache29": past_key_values[59],
                "key_cache30": past_key_values[60],
                "value_cache30": past_key_values[61],
                "key_cache31": past_key_values[62],
                "value_cache31": past_key_values[63],
            },
        )
        logits, new_past_key_values = output[0], output[1:]

    return logits, new_past_key_values


def logits_processor(scores):
    bad_words_id_length_1 = 3
    bad_words_mask = np.zeros(scores.shape[1])
    bad_words_mask[bad_words_id_length_1] = 1
    bad_words_mask = np.expand_dims(bad_words_mask, axis=0)

    scores = np.where(bad_words_mask, -np.inf, scores)

    return scores


def stopping_criteria(input_ids: np.array) -> np.array:
    max_length = 768
    cur_len = input_ids.shape[-1]
    is_done = cur_len >= max_length
    is_done = np.full(input_ids.shape[0], is_done)

    eos_token_id = np.array([151643])
    is_done = is_done | np.isin(input_ids[:, -1], eos_token_id)

    return is_done


def greedy_search(models, input_ids, last_hidden_state):
    eos_token_id = 2

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    unfinished_sequences = np.ones(batch_size, dtype=int)

    past_key_values = [np.zeros((1, 16, 0, 64), dtype=np.float16)] * 16

    blob_copy = False
    while True:
        # prepare model inputs
        if 0 < past_key_values[0].shape[1]:
            model_input_ids = input_ids[:, -1:]
        else:
            model_input_ids = input_ids
        pad_token_id = 1
        attention_mask = (input_ids != pad_token_id).astype(np.int64)

        if args.benchmark:
            start = int(round(time.time() * 1000))

        net = models["net"]
        logits, past_key_values = forward(
            net,
            model_input_ids,
            attention_mask,
            last_hidden_state,
            past_key_values,
            blob_copy,
        )
        blob_copy = True if COPY_BLOB_DATA else False

        if args.benchmark:
            end = int(round(time.time() * 1000))
            estimation_time = end - start
            logger.info(f"\tdecode time {estimation_time} ms")

        next_token_logits = logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(next_token_logits)

        # token selection
        next_tokens = np.argmax(next_token_scores, axis=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

        # update generated ids, model inputs, and length for next step
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)
        cur_len += 1

        unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id)
        if np.max(unfinished_sequences) == 0 or stopping_criteria(input_ids):
            break

    return input_ids


def predict(models, img):
    tokenizer = models["tokenizer"]

    last_hidden_state = np.load("last_hidden_state.npy")

    input_ids = np.array([[57579]])
    sequences = greedy_search(models, input_ids, last_hidden_state)

    output = []
    for seq in tokenizer.batch_decode(sequences):
        seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
        seq = re.sub(
            r"<.*?>", "", seq, count=1
        ).strip()  # remove first task start token
        output.append(token2json(tokenizer, seq))

    return output[0]


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
                output = predict(models, img)
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
            output = predict(models, img)

        print(json.dumps(output, indent=4, ensure_ascii=False))

    logger.info("Script finished successfully.")


def recognize_from_video(models):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), "Cannot capture source"

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
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = predict(models, img)

        # plot result
        res_img = draw_bbox(frame, out)

        # show
        cv2.imshow("frame", res_img)
        frame_shown = True

        # save results
        if writer is not None:
            res_img = res_img.astype(np.uint8)
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def main():
    # # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_XXX_PATH, MODEL_XXX_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("./tokenizer")
    else:
        raise NotImplementedError

    models = dict(
        tokenizer=tokenizer,
        net=net,
    )

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == "__main__":
    main()
