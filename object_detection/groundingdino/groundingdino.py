import sys
import time
import datetime

import numpy as np
import cv2
from PIL import Image
from transformers import AutoTokenizer

import ailia


# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from math_utils import sigmoid
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

# logger
from logging import getLogger  # noqa


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "groundingdino_swint_ogc.onnx"
MODEL_PATH = "groundingdino_swint_ogc.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/grounded-sam/"

IMAGE_PATH = "demo7.jpg"
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Grounded-SAM", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "--seed",
    type=int,
    default=int(datetime.datetime.now().strftime("%Y%m%d")),
    help="random seed for selection the color of the box",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="SwinB_896_4x",
    choices=("SwinB_896_4x", "R50_640_4x"),
    help="model type",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def generate_masks_with_special_tokens(input_ids, special_tokens_list):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (np.ndarray): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        np.ndarray: attention mask between each special tokens.
    """
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = np.eye(num_token, dtype=bool)[None, ...].repeat(bs, axis=0)
    position_ids = np.zeros((bs, num_token), dtype=int)
    previous_col = 0
    for i in range(len(idxs[0])):
        row, col = idxs[0][i], idxs[1][i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[
                row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
            ] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(
                0,
                col - previous_col,
            )
            c2t_maski = np.zeros((num_token), dtype=bool)
            c2t_maski[previous_col + 1 : col] = True
        previous_col = col

    return attention_mask, position_ids


def draw_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    logits: np.ndarray,
    phrases: list,
) -> np.ndarray:
    print(boxes)
    print(logits)
    print(phrases)
    # h, w, _ = image.shape
    # boxes = boxes * torch.Tensor([w, h, w, h])
    # xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # detections = sv.Detections(xyxy=xyxy)

    # labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # annotated_frame = box_annotator.annotate(
    #     scene=annotated_frame, detections=detections, labels=labels
    # )
    # return annotated_frame


# ======================
# Main functions
# ======================


def preprocess(img):
    return img


def post_processing(tokenizer, caption, pred_logits, pred_boxes):
    prediction_logits = sigmoid(pred_logits.astype(np.float32))[0]
    prediction_boxes = pred_boxes[0]

    box_threshold = 0.35
    mask = np.max(prediction_logits, axis=1) > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    text_threshold = 0.25
    # get_phrases_from_posmap
    tokenized = tokenizer(caption)
    non_zero_idx = [np.nonzero(logit > text_threshold)[0].tolist() for logit in logits]
    token_ids_list = [[tokenized["input_ids"][i] for i in li] for li in non_zero_idx]
    phrases = [tokenizer.decode(token_ids) for token_ids in token_ids_list]

    logits = np.max(logits, axis=1)

    return boxes, logits, phrases


def predict(models, img, caption):
    tokenizer = models["tokenizer"]

    captions = [caption]
    tokenized = tokenizer(captions, padding="longest", return_tensors="np")

    (text_self_attention_masks, position_ids) = generate_masks_with_special_tokens(
        tokenized["input_ids"], tokenizer.specical_tokens
    )

    max_text_len = 256
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, :max_text_len, :max_text_len
        ]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    # extract text embeddings
    tokenized_for_encoder = {
        k: v for k, v in tokenized.items() if k != "attention_mask"
    }
    tokenized_for_encoder["attention_mask"] = text_self_attention_masks
    tokenized_for_encoder["position_ids"] = position_ids

    input_ids = tokenized_for_encoder["input_ids"]
    token_type_ids = tokenized_for_encoder["token_type_ids"]
    attention_mask = tokenized_for_encoder["attention_mask"]
    position_ids = tokenized_for_encoder["position_ids"]
    text_token_mask = tokenized.attention_mask.astype(bool)

    # feedforward
    net = models["net"]
    if not args.onnx:
        output = net.predict(
            [
                samples,
                input_ids,
                token_type_ids,
                attention_mask,
                position_ids,
                text_token_mask,
            ]
        )
    else:
        output = net.run(
            None,
            {
                "samples": samples,
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "text_token_mask": text_token_mask,
            },
        )
    pred_logits, pred_boxes = output

    boxes, logits, phrases = post_processing(
        tokenizer, caption, pred_logits, pred_boxes
    )

    return boxes, logits, phrases


def recognize_from_image(models):
    caption = "horse. clouds. grasses. sky. hill."

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
                output = predict(models, img, caption)
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
            output = predict(models, img, caption)

        boxes, logits, phrases = output

        logger.info("detected %d instances" % len(boxes))

        # draw prediction
        res_img = draw_predictions(img, boxes, logits, phrases)

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

        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    tokenizer.specical_tokens = tokenizer.convert_tokens_to_ids(
        ["[CLS]", "[SEP]", ".", "?"]
    )

    models = {
        "tokenizer": tokenizer,
        "net": net,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
