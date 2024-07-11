import sys
import time

import numpy as np
import cv2
from PIL import Image
from transformers import AutoTokenizer

import ailia


# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from math_utils import sigmoid
from detector_utils import load_image, plot_results, hsv_to_rgb  # noqa

# logger
from logging import getLogger  # noqa


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_GDINO_PATH = "groundingdino_swint_ogc.onnx"
MODEL_GDINO_PATH = "groundingdino_swint_ogc.onnx.prototxt"
WEIGHT_SAM_PATH = "sam_vit_h_4b8939.onnx"
MODEL_SAM_PATH = "sam_vit_h_4b8939.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/grounded-sam/"

IMAGE_PATH = "demo.jpg"
SAVE_IMAGE_PATH = "output.png"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Grounded-SAM", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "--caption",
    type=str,
    default="The running dog.",
    help="Text prompt.",
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


def preprocess_shape(h, w, long_side_length):
    scale = long_side_length * 1.0 / max(h, w)
    new_h, new_w = h * scale, w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)

    return new_h, new_w


def mask_annotate(img, masks):
    colored_mask = np.array(img, copy=True, dtype=np.uint8)
    area = np.array([np.sum(mask) for mask in masks])
    for detection_idx in np.flip(np.argsort(area)):
        mask = masks[detection_idx]
        colored_mask[mask] = (255, 255, 255)

    opacity = 0.5
    img = cv2.addWeighted(colored_mask, opacity, img, 1 - opacity, 0)

    return img.astype(np.uint8)


# ======================
# Main functions
# ======================


def preprocess(img):
    im_h, im_w, _ = img.shape

    # Resize
    size = 800
    max_size = 1333
    min_original_size = min(im_w, im_h)
    max_original_size = max(im_w, im_h)
    if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

    if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
        oh, ow = im_h, im_w
    elif im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)

    img = np.asarray(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    # Normalize
    img = normalize_image(img, normalize_type="ImageNet")

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float16)

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


def predict_grounding_dino(models, img, caption):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)

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
    net = models["grounding_dino"]
    if not args.onnx:
        output = net.predict(
            [
                img,
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
                "samples": img,
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


def segment(net, img, xyxy):
    img = img[:, :, ::-1]  # BGR -> RGB
    original_size = img.shape[:2]

    target_length = 1024
    target_size = preprocess_shape(original_size[0], original_size[1], target_length)
    img = np.array(
        Image.fromarray(img).resize(target_size[::-1], Image.Resampling.BILINEAR)
    )

    img = normalize_image(img, normalize_type="ImageNet")

    img_size = 1024
    pad_h = img_size - target_size[0]
    pad_w = img_size - target_size[1]
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    result_masks = []
    for box in xyxy:
        # Transform
        new_h, new_w = preprocess_shape(
            original_size[0], original_size[1], target_length
        )

        coords = box.reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] * (new_w / original_size[1])
        coords[..., 1] = coords[..., 1] * (new_h / original_size[0])
        box = box.reshape(-1, 4)
        box = box[None, :]

        # feedforward
        input_size = np.array(target_size, dtype=int)
        original_size = np.array(original_size, dtype=int)
        if not args.onnx:
            output = net.predict([img, box, input_size, original_size])
        else:
            output = net.run(
                None,
                {
                    "image": img,
                    "box": box,
                    "input_size": input_size,
                    "original_size": original_size,
                },
            )
        masks, iou_predictions, low_res_masks = output

        masks = masks[0]
        scores = iou_predictions[0]
        logits = low_res_masks[0]
        index = np.argmax(scores)
        result_masks.append(masks[index])

    return np.array(result_masks)


def predict(models, img, caption):
    height, width, _ = img.shape
    boxes, logits, phrases = predict_grounding_dino(models, img, caption)

    boxes = boxes * np.array([width, height, width, height], dtype=np.float32)

    cx, cy, w, h = np.split(boxes, 4, axis=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    xyxy = np.concatenate((x1, y1, x2, y2), axis=-1)

    net = models["sam"]
    mask = segment(net, img, xyxy)

    return mask


def recognize_from_image(models):
    caption = args.caption
    if not caption.endswith("."):
        caption = caption + "."

    logger.info("Caption: " + caption)

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

        mask = output
        res_img = mask_annotate(img, mask)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        cv2.imwrite(savepath, res_img)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_GDINO_PATH, MODEL_GDINO_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SAM_PATH, MODEL_SAM_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        grounding_dino = ailia.Net(MODEL_GDINO_PATH, WEIGHT_GDINO_PATH, env_id=env_id)
        net = ailia.Net(MODEL_GDINO_PATH, WEIGHT_GDINO_PATH, env_id=env_id)
    else:
        import onnxruntime

        grounding_dino = onnxruntime.InferenceSession(WEIGHT_GDINO_PATH)
        net = onnxruntime.InferenceSession(WEIGHT_SAM_PATH)

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    tokenizer.specical_tokens = tokenizer.convert_tokens_to_ids(
        ["[CLS]", "[SEP]", ".", "?"]
    )

    models = {
        "tokenizer": tokenizer,
        "grounding_dino": grounding_dino,
        "sam": net,
    }

    recognize_from_image(models)


if __name__ == "__main__":
    main()
