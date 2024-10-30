import re

import numpy as np


parse_tasks_configs = {
    "od": {
        "TASK_NAME": "od",
        "PATTERN": "([a-zA-Z0-9 ]+)<loc_(\\\\d+)><loc_(\\\\d+)><loc_(\\\\d+)><loc_(\\\\d+)>",
    },
    "ocr": {
        "TASK_NAME": "ocr",
        "PATTERN": "(.+?)<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>",
        "AREA_THRESHOLD": 0.0,
    },
    "phrase_grounding": {"TASK_NAME": "phrase_grounding", "FILTER_BY_BLACK_LIST": True},
    "pure_text": {"TASK_NAME": "pure_text"},
    "description_with_bboxes": {"TASK_NAME": "description_with_bboxes"},
    # "description_with_polygons": {"TASK_NAME": "description_with_polygons"},
    # "polygons": {"TASK_NAME": "polygons"},
    "bboxes": {"TASK_NAME": "bboxes"},
    # "description_with_bboxes_or_polygons": {
    #     "TASK_NAME": "description_with_bboxes_or_polygons"
    # },
}

black_list_of_phrase_grounding = {
    # fmt: off
    "yours", "whom", "myself", "its", "everyone", "an", "some", "her", "ourselves", "itself", 
    "ours", "mine", "nobody", "one", "he", "yourselves", "several", "all", "lots", "we", 
    "one another", "each", "yourself", "anybody", "theirs", "a", "him", "the", "another", 
    "those", "hers", "many", "what", "who", "somebody", "their", "everything", "whose", 
    "other objects", "each other", "a set", "themselves", "which", "our", "a group", "anything", 
    "you", "this", "none", "it", "that", "she", "himself", "someone", "these", "us", "your", 
    "I", "images", "any", "oneself", "herself", "anyone", "them", "his", "few", "something", 
    "they", "the image", "everybody", "me", "image"
    # fmt: on
}


class BoxQuantizer(object):
    def __init__(self, bins):
        self.bins = bins

    def dequantize(self, boxes: np.ndarray, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = np.split(
            boxes, boxes.shape[-1], axis=-1
        )  # Shape: 4 * [N, 1].

        # Add 0.5 to use the center position of the bin as the coordinate.
        dequantized_xmin = (xmin + 0.5) * size_per_bin_w
        dequantized_ymin = (ymin + 0.5) * size_per_bin_h
        dequantized_xmax = (xmax + 0.5) * size_per_bin_w
        dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        dequantized_boxes = np.concatenate(
            (dequantized_xmin, dequantized_ymin, dequantized_xmax, dequantized_ymax),
            axis=-1,
        )

        return dequantized_boxes


box_quantizer = BoxQuantizer((1000, 1000))


class CoordinatesQuantizer(object):
    """
    Quantize coornidates (Nx2)
    """

    def __init__(self, bins):
        self.bins = bins

    def dequantize(self, coordinates: np.ndarray, size):
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size  # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        x, y = np.split(coordinates, 2, axis=-1)  # Shape: 4 * [N, 1].

        # Add 0.5 to use the center position of the bin as the coordinate.
        dequantized_x = (x + 0.5) * size_per_bin_w
        dequantized_y = (y + 0.5) * size_per_bin_h

        dequantized_coordinates = np.concatenate(
            (dequantized_x, dequantized_y), axis=-1
        )

        return dequantized_coordinates


coordinates_quantizer = CoordinatesQuantizer((1000, 1000))


def parse_ocr_from_text_and_spans(
    text,
    pattern,
    image_size,
    area_threshold=-1.0,
):
    bboxes = []
    labels = []
    text = text.replace("<s>", "")
    # ocr with regions
    parsed = re.findall(pattern, text)
    instances = []
    image_width, image_height = image_size

    for ocr_line in parsed:
        ocr_content = ocr_line[0]
        quad_box = ocr_line[1:]
        quad_box = [int(i) for i in quad_box]
        quad_box = (
            coordinates_quantizer.dequantize(
                np.array(quad_box).reshape(-1, 2), size=image_size
            )
            .reshape(-1)
            .tolist()
        )

        if area_threshold > 0:
            x_coords = [i for i in quad_box[0::2]]
            y_coords = [i for i in quad_box[1::2]]

            # apply the Shoelace formula
            area = 0.5 * abs(
                sum(
                    x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
                    for i in range(4 - 1)
                )
            )

            if area < (image_width * image_height) * area_threshold:
                continue

        bboxes.append(quad_box)
        labels.append(ocr_content)
        instances.append(
            {
                "quad_box": quad_box,
                "text": ocr_content,
            }
        )
    return instances


def parse_phrase_grounding_from_text_and_spans(text, pattern, image_size):
    # ignore <s> </s> and <pad>
    cur_span = 0
    if text.startswith("<s>"):
        cur_span += 3

    text = text.replace("<s>", "")
    text = text.replace("</s>", "")
    text = text.replace("<pad>", "")

    pattern = r"([^<]+(?:<loc_\d+>){4,})"
    phrases = re.findall(pattern, text)

    # pattern should be text pattern and od pattern
    pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
    box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

    instances = []
    for pharse_text in phrases:
        phrase_text_strip = pharse_text.replace("<ground>", "", 1)
        phrase_text_strip = pharse_text.replace("<obj>", "", 1)

        if phrase_text_strip == "":
            cur_span += len(pharse_text)
            continue

        # Prepare instance.
        instance = {}

        # parse phrase, get string
        phrase = re.search(pattern, phrase_text_strip)
        if phrase is None:
            cur_span += len(pharse_text)
            continue

        # parse bboxes by box_pattern
        bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
        if len(bboxes_parsed) == 0:
            cur_span += len(pharse_text)
            continue

        phrase = phrase.group()
        # remove leading and trailing spaces
        phrase = phrase.strip()

        if phrase in black_list_of_phrase_grounding:
            cur_span += len(pharse_text)
            continue

        # a list of list
        bbox_bins = [
            [int(_bboxes_parsed.group(j)) for j in range(1, 5)]
            for _bboxes_parsed in bboxes_parsed
        ]
        instance["bbox"] = box_quantizer.dequantize(
            boxes=np.array(bbox_bins), size=image_size
        ).tolist()

        # exclude non-ascii characters
        phrase = phrase.encode("ascii", errors="ignore").decode("ascii")
        instance["cat_name"] = phrase

        instances.append(instance)

    return instances


def parse_description_with_bboxes_from_text_and_spans(
    text, pattern, image_size, allow_empty_phrase=False
):
    # temporary parse solution, split by '.'
    # ignore <s> </s> and <pad>

    text = text.replace("<s>", "")
    text = text.replace("</s>", "")
    text = text.replace("<pad>", "")

    if allow_empty_phrase:
        pattern = rf"(?:(?:<loc_\d+>){{4,}})"
    else:
        pattern = r"([^<]+(?:<loc_\d+>){4,})"
    phrases = re.findall(pattern, text)

    # pattern should be text pattern and od pattern
    pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
    box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

    instances = []
    for pharse_text in phrases:
        phrase_text_strip = pharse_text.replace("<ground>", "", 1)
        phrase_text_strip = pharse_text.replace("<obj>", "", 1)

        if phrase_text_strip == "" and not allow_empty_phrase:
            continue

        # parse phrase, get string
        phrase = re.search(pattern, phrase_text_strip)
        if phrase is None:
            continue

        phrase = phrase.group()
        # remove leading and trailing spaces
        phrase = phrase.strip()

        # parse bboxes by box_pattern
        bboxes_parsed = list(re.finditer(box_pattern, pharse_text))
        if len(bboxes_parsed) == 0:
            continue

        # a list of list
        bbox_bins = [
            [int(_bboxes_parsed.group(j)) for j in range(1, 5)]
            for _bboxes_parsed in bboxes_parsed
        ]

        bboxes = box_quantizer.dequantize(
            boxes=np.array(bbox_bins), size=image_size
        ).tolist()

        phrase = phrase.encode("ascii", errors="ignore").decode("ascii")
        for _bboxes in bboxes:
            # Prepare instance.
            instance = {}
            instance["bbox"] = _bboxes
            # exclude non-ascii characters
            instance["cat_name"] = phrase
            instances.append(instance)

    return instances


def post_processor(
    text=None,
    image_size=None,
    task_answer_type=None,
):
    parsed_dict = {"text": text}

    parse_tasks = [
        "od",
        "ocr",
        "phrase_grounding",
        "pure_text",
        "description_with_bboxes",
        # "description_with_polygons",
        # "polygons",
        "bboxes",
        # "description_with_bboxes_or_polygons",
    ]
    for task in parse_tasks:
        if parse_tasks is not None and task not in task_answer_type:
            continue

        pattern = parse_tasks_configs[task].get("PATTERN", None)

        if task == "ocr":
            instances = parse_ocr_from_text_and_spans(
                text, pattern=pattern, image_size=image_size, area_threshold=0.0
            )
            parsed_dict["ocr"] = instances
        elif task == "phrase_grounding":
            instances = parse_phrase_grounding_from_text_and_spans(
                text,
                pattern=pattern,
                image_size=image_size,
            )
            parsed_dict["phrase_grounding"] = instances
        elif task == "pure_text":
            parsed_dict["pure_text"] = text
        elif task == "description_with_bboxes":
            instances = parse_description_with_bboxes_from_text_and_spans(
                text,
                pattern=pattern,
                image_size=image_size,
            )
            parsed_dict["description_with_bboxes"] = instances
        # elif task == "description_with_polygons":
        #     instances = self.parse_description_with_polygons_from_text_and_spans(
        #         text,
        #         pattern=pattern,
        #         image_size=image_size,
        #     )
        #     parsed_dict["description_with_polygons"] = instances
        # elif task == "polygons":
        #     instances = self.parse_description_with_polygons_from_text_and_spans(
        #         text,
        #         pattern=pattern,
        #         image_size=image_size,
        #         allow_empty_phrase=True,
        #     )
        #     parsed_dict["polygons"] = instances
        elif task == "bboxes":
            instances = parse_description_with_bboxes_from_text_and_spans(
                text,
                pattern=pattern,
                image_size=image_size,
                allow_empty_phrase=True,
            )
            parsed_dict["bboxes"] = instances
        # elif task == "description_with_bboxes_or_polygons":
        #     if "<poly>" in text:
        #         # only support either polygons or bboxes, not both at the same time
        #         instances = self.parse_description_with_polygons_from_text_and_spans(
        #             text,
        #             pattern=pattern,
        #             image_size=image_size,
        #         )
        #     else:
        #         instances = self.parse_description_with_bboxes_from_text_and_spans(
        #             text,
        #             pattern=pattern,
        #             image_size=image_size,
        #         )
        #     parsed_dict["description_with_bboxes_or_polygons"] = instances
        else:
            raise ValueError("task {} is not supported".format(task))

    return parsed_dict


def post_process_generation(text, task, image_size):
    """
    Post-process the output of the model to each of the task outputs.
    """

    tasks_answer_post_processing_type = {
        "<OCR>": "pure_text",
        "<OCR_WITH_REGION>": "ocr",
        "<CAPTION>": "pure_text",
        "<DETAILED_CAPTION>": "pure_text",
        "<MORE_DETAILED_CAPTION>": "pure_text",
        "<OD>": "description_with_bboxes",
        "<DENSE_REGION_CAPTION>": "description_with_bboxes",
        "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
        # "<REFERRING_EXPRESSION_SEGMENTATION>": "polygons",
        # "<REGION_TO_SEGMENTATION>": "polygons",
        # "<OPEN_VOCABULARY_DETECTION>": "description_with_bboxes_or_polygons",
        # "<REGION_TO_CATEGORY>": "pure_text",
        # "<REGION_TO_DESCRIPTION>": "pure_text",
        # "<REGION_TO_OCR>": "pure_text",
        "<REGION_PROPOSAL>": "bboxes",
    }

    task_answer_type = tasks_answer_post_processing_type.get(task, "pure_text")
    task_answer = post_processor(
        text=text,
        image_size=image_size,
        task_answer_type=task_answer_type,
    )[task_answer_type]

    if task_answer_type == "pure_text":
        final_answer = task_answer
        # remove the special tokens
        final_answer = final_answer.replace("<s>", "").replace("</s>", "")
    elif task_answer_type in [
        "od",
        "description_with_bboxes",
        "bboxes",
    ]:
        od_instances = task_answer
        bboxes_od = [_od_instance["bbox"] for _od_instance in od_instances]
        labels_od = [str(_od_instance["cat_name"]) for _od_instance in od_instances]
        final_answer = {"bboxes": bboxes_od, "labels": labels_od}
    elif task_answer_type in ["ocr"]:
        bboxes = [_od_instance["quad_box"] for _od_instance in task_answer]
        labels = [str(_od_instance["text"]) for _od_instance in task_answer]
        final_answer = {"quad_boxes": bboxes, "labels": labels}
    elif task_answer_type in ["phrase_grounding"]:
        bboxes = []
        labels = []
        for _grounded_phrase in task_answer:
            for _bbox in _grounded_phrase["bbox"]:
                bboxes.append(_bbox)
                labels.append(_grounded_phrase["cat_name"])
        final_answer = {"bboxes": bboxes, "labels": labels}
    # elif task_answer_type in ["description_with_polygons", "polygons"]:
    #     labels = []
    #     polygons = []
    #     for result in task_answer:
    #         label = result["cat_name"]
    #         _polygons = result["polygons"]
    #         labels.append(label)
    #         polygons.append(_polygons)
    #     final_answer = {"polygons": polygons, "labels": labels}
    # elif task_answer_type in ["description_with_bboxes_or_polygons"]:
    #     bboxes = []
    #     bboxes_labels = []
    #     polygons = []
    #     polygons_labels = []
    #     for result in task_answer:
    #         label = result["cat_name"]
    #         if "polygons" in result:
    #             _polygons = result["polygons"]
    #             polygons.append(_polygons)
    #             polygons_labels.append(label)
    #         else:
    #             _bbox = result["bbox"]
    #             bboxes.append(_bbox)
    #             bboxes_labels.append(label)
    #     final_answer = {
    #         "bboxes": bboxes,
    #         "bboxes_labels": bboxes_labels,
    #         "polygons": polygons,
    #         "polygons_labels": polygons_labels,
    #     }
    else:
        raise ValueError(
            "Unknown task answer post processing type: {}".format(task_answer_type)
        )

    final_answer = {task: final_answer}
    return final_answer
