from datasets_detic.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from datasets_detic.in21k_categories import IN21K_CATEGORIES


def get_lvis_meta_v1():
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}

    return meta


def get_in21k_meta_v1():
    thing_classes = IN21K_CATEGORIES
    meta = {"thing_classes": thing_classes}

    return meta
