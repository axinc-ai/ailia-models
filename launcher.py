# ailia MODELS launcher

import os
import cv2
import numpy
import subprocess
import shutil
import sys
from PIL import Image, ImageTk

# for macOS, please install "brew install python-tk@3.9"
import tkinter as tk
from tkinter import ttk

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
    "illustration2vec", "etl", "vggface2", "anomaly_detection", "natural_language_processing", "audio_processing"
]

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

    if ("natural_language_processing" == model["category"]) or \
        ("audio_processing" == model["category"]):
        args_dict["video"]=None
    else:
        if not args_dict["video"]:
            args_dict["video"]=0

    options = []
    for key in args_dict:
        if key=="ftype":
            continue
        if args_dict[key] is not None:
            if args_dict[key] is True:
                options.append("--"+key)
            elif args_dict[key] is False:
                continue
            else:
                options.append("--"+key)
                options.append(str(args_dict[key]))
    
    cmd = [cmd, model["model"]+".py"] + options
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=dir, shell=False)


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


ListboxModel = None
canvas = None
canvas_item = None

def load_image(path):
    print(path)
    global canvas, canvas_item
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr,(320,240))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
    image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
    if canvas_item == None:
        canvas_item = canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
    else:
        canvas.itemconfig(canvas_item,image=image_tk)


def search():
    global model_list
    print(model_list)

    global ListboxModel
    selected = [ int(x) for x in ListboxModel.curselection()]
    print("search "+str(selected))

    if selected==[]:
        selected=[0]

    model_request=model_list[int(selected[0])]
    open_model(model_request)

def get_camera_list():
    return [0]

    print("List cameras")
    index = 0
    cameras = []
    while True:
        print(index)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append(index)
        else:
            break
        index=index+1
        cap.release()
    return cameras

def model_changed(event):
    selection = event.widget.curselection()
    if selection:
        index = selection[0]
        model_request = model_list[index]
        for ext in [".jpg",".png"]:
            image_path = "./"+model_request["category"]+"/"+model_request["model"]+"/output"+ext
            if os.path.exists(image_path):
                load_image(image_path)
                print(image_path)
                print(index)
                break

def main():
    global ListboxModel
    global model_list, model_request, model_loading_cnt, invalidate_quit_cnt
    global canvas

    model_list, category_cnt = search_model()

    #WINDOW_COL = int((len(model_list)+WINDOW_ROW-1)/WINDOW_ROW)

    #window_width = (BUTTON_WIDTH + BUTTON_MARGIN) * WINDOW_COL
    #window_height = (BUTTON_HEIGHT + BUTTON_MARGIN) * WINDOW_ROW

    #img = numpy.zeros((window_height, window_width, 3)).astype(numpy.uint8)

    # rootメインウィンドウの設定
    root = tk.Tk()
    root.title("ailia MODELS")
    root.geometry("800x600")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20,pady=10)

    # Listboxの選択肢
    models = []
    for i in range(len(model_list)):
        models.append(""+model_list[i]["category"]+" : "+model_list[i]["model"])

    lists = tk.StringVar(value=models)
    listsCamera = tk.StringVar(value=get_camera_list())

    # 各種ウィジェットの作成
    ListboxModel = tk.Listbox(frame, listvariable=lists, width=50, height=20, selectmode="single")
    ListboxCamera = tk.Listbox(frame, listvariable=listsCamera, width=50, height=2, selectmode="single")

    ListboxModel.bind("<<ListboxSelect>>", model_changed)

    # スクロールバーの作成
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=ListboxModel.yview)

    # スクロールバーをListboxに反映
    ListboxModel["yscrollcommand"] = scrollbar.set

    # StringVarのインスタンスを格納する変数textの設定
    text = tk.StringVar(frame)
    text.set("Run")

    textModel = tk.StringVar(frame)
    textModel.set("Models")

    textCamera = tk.StringVar(frame)
    textCamera.set("Cameras")

    textPreview = tk.StringVar(frame)
    textPreview.set("Preview")

    textEnvironment = tk.StringVar(frame)
    textEnvironment.set("Environment")

    # 各種ウィジェットの作成
    labelModel = tk.Label(frame, textvariable=textModel)
    labelCamera = tk.Label(frame, textvariable=textCamera)
    labelPreview = tk.Label(frame, textvariable=textPreview)
    labelEnvironment = tk.Label(frame, textvariable=textEnvironment)

    button = tk.Button(frame, textvariable=text, command=search, width=10)




    canvas = tk.Canvas(frame, bg="black", width=320, height=240)
    canvas.place(x=0, y=0)
    load_image("./object_detection/yolox/output.jpg")

    # 各種ウィジェットの設置
    labelModel.grid(row=0, column=0, sticky=tk.NW)
    ListboxModel.grid(row=1, column=0, sticky=tk.NW, rowspan = 2)
    scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S), rowspan = 2)

    labelPreview.grid(row=0, column=2, sticky=tk.NW)
    canvas.grid(row=1, column=2, sticky=tk.NW)

    labelCamera.grid(row=0, column=3, sticky=tk.NW)
    ListboxCamera.grid(row=1, column=3, sticky=tk.NW)

    button.grid(row=4, column=0, sticky=tk.NW)



    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20, pady=10)



    root.mainloop()


    #cv2.imshow('ailia MODELS', img)
    #cv2.setMouseCallback("ailia MODELS", mouse_callback)

    #while(True):
    #    if cv2.waitKey(1) & 0xFF == ord('q') and invalidate_quit_cnt<=0:
    #        break

    #    if model_request is not None and model_loading_cnt<=0:
    #        open_model(model_request)
    #        model_request=None
    #        invalidate_quit_cnt=10
    #        click_trig=False
    #        continue

    #    if model_request is not None:
    #        display_loading(img, model_request)
    #        model_loading_cnt = model_loading_cnt - 1
    #    else:
    #        display_ui(img, model_list, category_cnt, window_width, window_height)
    #        invalidate_quit_cnt = invalidate_quit_cnt -1

    #    cv2.imshow('ailia MODELS', img)


    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



