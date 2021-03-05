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

BUTTON_WIDTH = 400
BUTTON_HEIGHT = 20
BUTTON_MARGIN = 2

WINDOW_ROW = 22

# ======================
# Model search
# ======================

IGNORE_LIST = [
    "commercial_model", "validation", ".git", "log", "prnet", "bert",
    "illustration2vec", "etl", "vggface2", ""
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


def display_ui(img, model_list, category_cnt, window_width, window_height):
    global mx, my, click_trig

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
                dir = "./"+model["category"]+"/"+model["model"]+"/"
                cmd = sys.executable
                if ("neural_language_processing" == model["category"]) or \
                   ("audio_processing" == model["category"]):
                    options = ""
                else:
                    video_id = args.video
                    if not args.video:
                        video_id = 0
                    options = "-v "+str(video_id)
                
                cmd = cmd + " " + model["model"]+".py" + " " + options
                
                subprocess.check_call(cmd, cwd=dir, shell=True)
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
    model_list, category_cnt = search_model()

    WINDOW_COL = int((len(model_list)+WINDOW_ROW-1)/WINDOW_ROW)

    window_width = (BUTTON_WIDTH + BUTTON_MARGIN) * WINDOW_COL
    window_height = (BUTTON_HEIGHT + BUTTON_MARGIN) * WINDOW_ROW

    img = numpy.zeros((window_height, window_width, 3)).astype(numpy.uint8)

    cv2.imshow('ailia MODELS', img)
    cv2.setMouseCallback("ailia MODELS", mouse_callback)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        display_ui(img, model_list, category_cnt, window_width, window_height)
        cv2.imshow('ailia MODELS', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
