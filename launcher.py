# ailia MODELS launcher

import os
import cv2
import numpy
import subprocess
import shutil
import sys

sys.path.append('./util')
from utils import get_base_parser, update_parser  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('ailia MODELS launcher', None, None)
args = update_parser(parser)


# ======================
# Settings
# ======================

BUTTON_WIDTH = 350
BUTTON_HEIGHT = 20
BUTTON_MARGIN = 2

WINDOW_ROW = 34

# ======================
# Model search
# ======================

IGNORE_LIST = [
    "commercial_model", "validation", ".git", "log", "prnet", "bert",
    "illustration2vec", "etl", "vggface2", "anomaly_detection"
]

try:
    import transformers
except ModuleNotFoundError:
    IGNORE_LIST.append("neural_language_processing")
    pass

try:
    import torchaudio
except ModuleNotFoundError:
    IGNORE_LIST.append("audio_processing")
    pass

def search_model():
    file_list = []
    for current, subfolders, subfiles in os.walk("./"):
        file_list.append(current)

    file_list.sort()

    model_list = []
    category_list = {}
    model_exist = {}
    for current in file_list:
        current = current.replace("\\", "/")
        files = current.split("/")
        if len(files) == 3:
            if (files[1] in IGNORE_LIST) or (files[2] in IGNORE_LIST):
                continue
            if files[2] in model_exist:
                continue
            script = "./"+files[1]+"/"+files[2]+"/"+files[2]+".py"
            if os.path.exists(script):
                if not(files[1] in category_list):
                    category_list[files[1]] = len(category_list)
                category_id = category_list[files[1]]
                model_list.append({
                    "category": files[1],
                    "category_id": category_id,
                    "model": files[2],
                })
                model_exist[files[2]] = True
    return model_list, len(category_list)


# ======================
# Model List
# ======================

mx = 0
my = 0
click_trig = False
model_request = None
model_loading_cnt = 0
invalidate_quit_cnt = 0

def mouse_callback(event, x, y, flags, param):
    global mx, my, click_trig
    if event == cv2.EVENT_LBUTTONDOWN:
        click_trig = True
    mx = x
    my = y


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        numpy.array([[[h, s, v]]], dtype=numpy.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def open_model(model):
    dir = "./"+model["category"]+"/"+model["model"]+"/"
    cmd = sys.executable

    args_dict = vars(args)

    if ("neural_language_processing" == model["category"]) or \
        ("audio_processing" == model["category"]):
        args_dict["video"]=None
    else:
        if not args_dict["video"]:
            args_dict["video"]=0

    options = ""
    for key in args_dict:
        if key=="ftype":
            continue
        if args_dict[key] is not None:
            if args_dict[key] is True:
                options = options + " --"+key
            elif args_dict[key] is False:
                continue
            else:
                options = options + " --"+key+" "+str(args_dict[key])
    
    cmd = cmd + " " + model["model"]+".py" + " " + options
    print(cmd)
    
    subprocess.check_call(cmd, cwd=dir, shell=True)


def display_loading(img, model):
    text = "Loading "+model["model"]

    fontScale = 0.75

    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0]
    tw = textsize[0]
    th = textsize[1]

    margin = 8

    top_left = ((img.shape[1] - tw)//2 - margin, (img.shape[0] - th)//2 - margin)
    bottom_right = (top_left[0] + tw + margin*2, top_left[1] + th + margin*2)
    
    color = (255,255,255,255)
    cv2.rectangle(img, top_left, bottom_right, color, thickness=-1)

    text_color = (0,0,0,255)
    cv2.putText(
        img,
        text,
        (top_left[0], top_left[1] + th + margin),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        text_color,
        1
    )


def display_ui(img, model_list, category_cnt, window_width, window_height):
    global mx, my, click_trig, model_request, model_loading_cnt

    cv2.rectangle(img, (0,0), (img.shape[1],img.shape[0]), (0,0,0,255), thickness=-1)

    x = BUTTON_MARGIN
    y = BUTTON_MARGIN
    w = BUTTON_WIDTH
    h = BUTTON_HEIGHT

    for model in model_list:
        color = hsv_to_rgb(
            256 * model["category_id"] / (category_cnt+1), 128, 255
        )

        if mx >= x and mx <= x+w and my >= y and my <= y+h:
            color = (255, 255, 255)
            if click_trig:
                model_request = model
                model_loading_cnt = 10
                click_trig = False

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)

        text_position = (x+4, y+int(BUTTON_HEIGHT/2)+4)

        color = (0, 0, 0)
        fontScale = 0.5

        cv2.putText(
            img,
            model["category"]+" : "+model["model"],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y = y + h + BUTTON_MARGIN

        if y >= window_height:
            y = BUTTON_MARGIN
            x = x + w + BUTTON_MARGIN

    click_trig = False


def main():
    global model_request, model_loading_cnt, invalidate_quit_cnt

    model_list, category_cnt = search_model()

    WINDOW_COL = int((len(model_list)+WINDOW_ROW-1)/WINDOW_ROW)

    window_width = (BUTTON_WIDTH + BUTTON_MARGIN) * WINDOW_COL
    window_height = (BUTTON_HEIGHT + BUTTON_MARGIN) * WINDOW_ROW

    img = numpy.zeros((window_height, window_width, 3)).astype(numpy.uint8)

    cv2.imshow('ailia MODELS', img)
    cv2.setMouseCallback("ailia MODELS", mouse_callback)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q') and invalidate_quit_cnt<=0:
            break

        if model_request is not None and model_loading_cnt<=0:
            open_model(model_request)
            model_request=None
            invalidate_quit_cnt=10
            click_trig=False
            continue

        if model_request is not None:
            display_loading(img, model_request)
            model_loading_cnt = model_loading_cnt - 1
        else:
            display_ui(img, model_list, category_cnt, window_width, window_height)
            invalidate_quit_cnt = invalidate_quit_cnt -1

        cv2.imshow('ailia MODELS', img)


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
