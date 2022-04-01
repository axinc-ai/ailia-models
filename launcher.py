# ailia MODELS launcher

import os
import cv2
import numpy
import subprocess
import shutil
import sys
import glob
import ailia

from PIL import Image, ImageTk

# for macOS, please install "brew install python-tk@3.9"
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

sys.path.append('./util')
from utils import get_base_parser, update_parser  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('ailia MODELS launcher', None, None)
args = update_parser(parser)

# ======================
# Global settings
# ======================

camera_index = 0
env_index = args.env_id
model_index = 0

# ======================
# Model search
# ======================

IGNORE_LIST = [
    "commercial_model", "validation", ".git", "log", "prnet", "bert",
    "illustration2vec", "etl", "vggface2", "anomaly_detection", "natural_language_processing", "audio_processing"
]

def get_model_list():
    global model_index

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


    model_name_list = []
    for i in range(len(model_list)):
        model_name_list.append(""+model_list[i]["category"]+" : "+model_list[i]["model"])
        if model_list[i]["model"]=="yolox":
            model_index = i

    return model_list, model_name_list, len(category_list)


# ======================
# Environment
# ======================

def get_camera_list():
    return ["Camera:0"]

    print("List cameras")
    index = 0
    cameras = []
    while True:
        print(index)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append("Camera:"+index)
        else:
            break
        index=index+1
        cap.release()
    return cameras

def camera_changed(event):
    global camera_index, camera_list
    selection = event.widget.curselection()
    if selection:
        camera_index = selection[0]
    else:
        camera_index = 0   
    print("camera",camera_index)

def get_env_list():
    env_list = []
    for env in ailia.get_environment_list():
        env_list.append(env.name)
    return env_list  

def environment_changed(event):
    global env_index
    selection = event.widget.curselection()
    if selection:
        env_index = selection[0]
    else:
        env_index = 0
    print("env",env_index)

# ======================
# Change model
# ======================


def model_changed(event):
    global model_index
    selection = event.widget.curselection()
    if selection:
        model_index = selection[0]
    else:
        model_index = 0
    load_detail(model_index)

def load_image(path):
    print(path)
    global canvas, canvas_item, image_tk
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr,(320,240))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
    image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
    if canvas_item == None:
        canvas_item = canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
    else:
        canvas.itemconfig(canvas_item,image=image_tk)

def load_detail(index):
    model_request = model_list[index]

    base_path =  "./"+model_request["category"]+"/"+model_request["model"]+"/"

    image_exist = False
    for ext in [".jpg",".png"]:
        image_path = base_path + "output" + ext
        if os.path.exists(image_path):
            load_image(image_path)
            image_exist = True
            break

    if not image_exist:
        files = glob.glob(base_path+"*.jpg")
        files.extend(glob.glob(base_path+"*.png"))
        for image_path in files:
            if os.path.exists(image_path):
                load_image(image_path)
                break

    global textModelDetail

    f = open(base_path+"README.md")
    text = f.readlines()
    text = text[0].replace("# ","")
    textModelDetail.set(text)


# ======================
# Run model
# ======================

proc = None

def open_model(model):
    global proc

    if not (proc==None):
        proc.kill()
        proc=None

    dir = "./"+model["category"]+"/"+model["model"]+"/"
    cmd = sys.executable

    args_dict = vars(args)

    if "Camera:" in camera_list[camera_index]:
        video_name = camera_index
    else:
        video_name = camera_list[camera_index]

    if not(("natural_language_processing" == model["category"]) or ("audio_processing" == model["category"])):
        if ".png" in str(video_name) or ".jpg" in str(video_name):
            if "video" in args_dict:
                del args_dict["video"]
            args_dict["input"]=video_name
            args_dict["savepath"]="temp.png"
        else:
            args_dict["video"]=video_name

    args_dict["env_id"]=env_index

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

    if not ("video" in args_dict):
        subprocess.check_call(cmd, cwd=dir, shell=False)
        model_request = model_list[model_index]
        load_image("./"+model_request["category"]+"/"+model_request["model"]+"/"+"temp.png")
    else:
        proc = subprocess.Popen(cmd, cwd=dir)
        outs, errs = proc.communicate(timeout=1)
    


def run_button_clicked():
    global model_list
    model_request=model_list[int(model_index)]
    open_model(model_request)


def stop_button_clicked():
    global proc

    if not (proc==None):
        proc.kill()
        proc=None


# ======================
# Select file
# ======================

def file_dialog():
    global listsCamera, ListboxCamera, camera_index
    fTyp = [("Image File or Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        camera_list.append(file_name)
        listsCamera.set(camera_list)
        ListboxCamera.select_clear(camera_index)
        camera_index = len(camera_list)-1
        ListboxCamera.select_set(camera_index)

# ======================
# GUI
# ======================

#ListboxModel = None
#textModelDetail = None

#canvas = None
canvas_item = None
#image_tk = None

def main():
    global ListboxModel, textModelDetail
    global model_list, model_request, model_loading_cnt, invalidate_quit_cnt
    global canvas, inputFile, listsCamera, camera_list, ListboxCamera

    model_list, model_name_list, category_cnt = get_model_list()

    # rootメインウィンドウの設定
    root = tk.Tk()
    root.title("ailia MODELS")
    root.geometry("800x600")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20,pady=10)

    # Listboxの選択肢
    env_list = get_env_list()
    camera_list = get_camera_list()

    lists = tk.StringVar(value=model_name_list)
    listsCamera = tk.StringVar(value=camera_list)
    listEnvironment =tk.StringVar(value=env_list)

    # 各種ウィジェットの作成
    ListboxModel = tk.Listbox(frame, listvariable=lists, width=40, height=20, selectmode="single", exportselection=False)
    ListboxCamera = tk.Listbox(frame, listvariable=listsCamera, width=40, height=4, selectmode="single", exportselection=False)
    ListboxEnvironment = tk.Listbox(frame, listvariable=listEnvironment, width=40, height=4, selectmode="single", exportselection=False)

    ListboxModel.bind("<<ListboxSelect>>", model_changed)
    ListboxCamera.bind("<<ListboxSelect>>", camera_changed)
    ListboxEnvironment.bind("<<ListboxSelect>>", environment_changed)

    ListboxModel.select_set(model_index)
    ListboxCamera.select_set(camera_index)
    ListboxEnvironment.select_set(env_index)

    # スクロールバーの作成
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=ListboxModel.yview)

    # スクロールバーをListboxに反映
    ListboxModel["yscrollcommand"] = scrollbar.set

    # StringVarのインスタンスを格納する変数textの設定
    textRun = tk.StringVar(frame)
    textRun.set("Run")

    textStop = tk.StringVar(frame)
    textStop.set("Stop")

    textInputFile = tk.StringVar(frame)
    textInputFile.set("Add File")

    textModel = tk.StringVar(frame)
    textModel.set("Models")

    textCamera = tk.StringVar(frame)
    textCamera.set("Input")

    textPreview = tk.StringVar(frame)
    textPreview.set("Please hold 'q' key for finish running model.")

    textEnvironment = tk.StringVar(frame)
    textEnvironment.set("Environment")

    textModelDetail = tk.StringVar(frame)
    textModelDetail.set("ModelDetail")

    # 各種ウィジェットの作成
    labelModel = tk.Label(frame, textvariable=textModel)
    labelCamera = tk.Label(frame, textvariable=textCamera)
    labelPreview = tk.Label(frame, textvariable=textPreview)
    labelEnvironment = tk.Label(frame, textvariable=textEnvironment)
    labelModelDetail = tk.Label(frame, textvariable=textModelDetail)

    buttonRun = tk.Button(frame, textvariable=textRun, command=run_button_clicked, width=10)
    buttonStop = tk.Button(frame, textvariable=textStop, command=stop_button_clicked, width=10)
    buttonInputFile = tk.Button(frame, textvariable=textInputFile, command=file_dialog, width=10)
    check = tk.Checkbutton(frame, text='Save results')

    canvas = tk.Canvas(frame, bg="black", width=320, height=240)
    canvas.place(x=0, y=0)
    load_detail(model_index)

    # 各種ウィジェットの設置
    labelModel.grid(row=0, column=0, sticky=tk.NW, columnspan=2)
    ListboxModel.grid(row=1, column=0, sticky=tk.NW, columnspan=2)
    scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))

    canvas.grid(row=1, column=3, sticky=tk.NW)
    labelModelDetail.grid(row=0, column=3, sticky=tk.NW)

    labelCamera.grid(row=3, column=0, sticky=tk.NW, columnspan=2)
    ListboxCamera.grid(row=4, column=0, sticky=tk.NW, columnspan=2)

    labelEnvironment.grid(row=3, column=3, sticky=tk.NW)
    ListboxEnvironment.grid(row=4, column=3, sticky=tk.NW)

    buttonRun.grid(row=2, column=0, sticky=tk.NW)
    buttonStop.grid(row=2, column=1, sticky=tk.NW)
    buttonInputFile.grid(row=6, column=0, sticky=tk.NW)
    #check.grid(row=2, column=2, sticky=tk.NW)

    #labelPreview.grid(row=7, column=0, sticky=tk.NW)


    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20, pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()



