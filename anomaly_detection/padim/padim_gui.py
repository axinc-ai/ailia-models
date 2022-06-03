# PaDiM GUI launcher

import os
import cv2
import numpy
import shutil
import sys
import glob
import ailia

from PIL import Image, ImageTk

sys.path.append('../../util')
from padim_utils import *
from model_utils import check_and_download_models  # noqa: E402

# for macOS, please install "brew install python-tk@3.9"
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

import log_init
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)


# ======================
# Global settings
# ======================

input_index = 0
output_index = 0
result_index = 0
slider_index = 50

# ======================
# Environment
# ======================

def input_changed(event):
    global input_index
    selection = event.widget.curselection()
    if selection:
        input_index = selection[0]
    else:
        input_index = 0   
    load_detail(train_list[input_index])

def output_changed(event):
    global output_index
    selection = event.widget.curselection()
    if selection:
        output_index = selection[0]
    else:
        output_index = 0   
    load_detail(test_list[output_index])

def result_changed(event):
    global result_index
    selection = event.widget.curselection()
    if selection:
        result_index = selection[0]
    else:
        result_index = 0   
    load_detail(result_list[result_index])

def slider_changed(event):
    global scale, slider_index
    slider_index = scale.get()

# ======================
# Change file
# ======================


def create_photo_image(path,w=320,h=320):
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr,(w,h))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
    image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    #image_pil.thumbnail((w,h), Image.ANTIALIAS)
    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
    return image_tk

def load_canvas_image(path):
    global canvas, canvas_item, image_tk
    image_tk = create_photo_image(path)
    if canvas_item == None:
        canvas_item = canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
    else:
        canvas.itemconfig(canvas_item,image=image_tk)

def load_detail(image_path):
    image_exist = False
    for ext in [".jpg",".png"]:
        if os.path.exists(image_path):
            load_canvas_image(image_path)
            image_exist = True
            break


# ======================
# Run model
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

def train_button_clicked():
    print("begin training")

    # model files check and download
    weight_path, model_path, params = get_params("resnet18")
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    # training
    batch_size = 32
    train_dir = "train"
    aug = False
    aug_num = 0
    seed = 1024
    train_outputs = training(net, params, batch_size, train_dir, aug, aug_num, seed, logger)

    # save learned distribution
    train_feat_file = "train.pkl"
    #train_dir = args.train_dir
    #train_feat_file = "%s.pkl" % os.path.basename(train_dir)
    logger.info('saving train set feature to: %s ...' % train_feat_file)
    with open(train_feat_file, 'wb') as f:
        pickle.dump(train_outputs, f)
    logger.info('saved.')

score_cache = {}

def test_button_clicked():
    global score_cache
    print("begin test")

    # model files check and download
    weight_path, model_path, params = get_params("resnet18")
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # create net instance
    env_id = ailia.get_gpu_environment_id()
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    # load trained model
    with open("train.pkl", 'rb') as f:
        train_outputs = pickle.load(f)

    # file loop
    test_imgs = []

    score_map = []
    for i_img in range(0, len(test_list)):
        logger.info('from (%s) ' % (test_list[i_img]))

        image_path = test_list[i_img]
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img)

        test_imgs.append(img[0])
        if image_path in score_cache:
            dist_tmp = score_cache[image_path].copy()
        else:
            dist_tmp = infer(net, params, train_outputs, img)
            score_cache[image_path] = dist_tmp.copy()
        score_map.append(dist_tmp)

    scores = normalize_scores(score_map)
    anormal_scores = calculate_anormal_scores(score_map)

    # Plot gt image
    os.makedirs("result", exist_ok=True)
    global result_list, listsResult, ListboxResult
    threshold = slider_index / 100.0
    for i in range(0, scores.shape[0]):
        img = denormalization(test_imgs[i])
        heat_map, mask, vis_img = visualize(img, scores[i], threshold)
        vis_img = (vis_img * 255).astype(np.uint8)
        path = test_list[i].split("/")
        output_path = "result/"+path[len(path)-1]
        cv2.imwrite(output_path, vis_img)       
        if not (output_path in result_list):
            result_list.append(output_path)

    listsResult.set(result_list)
    load_detail(result_list[0])
    ListboxResult.select_set(0)


# ======================
# Select file
# ======================

def train_file_dialog():
    global listsInput, ListboxInput, input_index
    fTyp = [("Image File or Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        input_list.append(file_name)
        listsInput.set(input_list)
        ListboxInput.select_clear(input_index)
        input_index = len(input_list)-1
        ListboxInput.select_set(input_index)

def test_file_dialog():
    global listsOutput, ListboxOutput, output_index
    fTyp = [("Image File or Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        output_list.append(file_name)
        listsOutput.set(output_list)
        ListboxOutput.select_clear(output_index)
        output_index = len(output_list)-1
        ListboxOutput.select_set(output_index)

def get_training_file_list():
    base_path = "train/"
    files = glob.glob(base_path+"*.jpg")
    files.extend(glob.glob(base_path+"*.png"))
    image_list = []
    for image_path in files:
        image_list.append(image_path)
    image_list.sort()
    return image_list

def get_test_file_list():
    return ["bottle_000.png"]

def get_result_file_list():
    return []

# ======================
# GUI
# ======================

canvas_item = None

def main():
    global train_list, test_list, result_list
    global listsResult, ListboxResult
    #global ListboxModel, textModelDetail
    #global model_list, model_request, model_loading_cnt, invalidate_quit_cnt
    global canvas, scale
    #global inputFile, listsInput, input_list, ListboxInput
    #global outputFile, listsOutput, output_list, ListboxOutput

    #model_list, model_name_list, category_cnt = get_model_list()

    # rootメインウィンドウの設定
    root = tk.Tk()
    root.title("PaDiM GUI")
    root.geometry("1200x600")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20,pady=10)

    # Listboxの選択肢
    train_list = get_training_file_list()
    test_list = get_test_file_list()
    result_list = get_result_file_list()

    listsInput = tk.StringVar(value=train_list)
    listsOutput = tk.StringVar(value=test_list)
    listsResult = tk.StringVar(value=result_list)

    # 各種ウィジェットの作成
    ListboxInput = tk.Listbox(frame, listvariable=listsInput, width=20, height=12, selectmode="single", exportselection=False)
    ListboxOutput = tk.Listbox(frame, listvariable=listsOutput, width=20, height=12, selectmode="single", exportselection=False)
    ListboxResult = tk.Listbox(frame, listvariable=listsResult, width=20, height=12, selectmode="single", exportselection=False)

    ListboxInput.bind("<<ListboxSelect>>", input_changed)
    ListboxOutput.bind("<<ListboxSelect>>", output_changed)
    ListboxResult.bind("<<ListboxSelect>>", result_changed)

    ListboxInput.select_set(input_index)
    ListboxOutput.select_set(output_index)
    ListboxResult.select_set(result_index)

    # スクロールバーの作成
    #scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=ListboxInput.yview)

    # スクロールバーをListboxに反映
    #ListboxInput["yscrollcommand"] = scrollbar.set

    # StringVarのインスタンスを格納する変数textの設定
    textRun = tk.StringVar(frame)
    textRun.set("Train")

    textStop = tk.StringVar(frame)
    textStop.set("Test")

    textInputFile = tk.StringVar(frame)
    textInputFile.set("Select train folder")

    textOutputFile = tk.StringVar(frame)
    textOutputFile.set("Select test folder")

    textInput = tk.StringVar(frame)
    textInput.set("Train images")

    textOutput = tk.StringVar(frame)
    textOutput.set("Test images")

    textResult = tk.StringVar(frame)
    textResult.set("Result images")

    textModelDetail = tk.StringVar(frame)
    textModelDetail.set("Preview")

    textSlider = tk.StringVar(frame)
    textSlider.set("Threshold")

    # 各種ウィジェットの作成
    labelInput = tk.Label(frame, textvariable=textInput)
    labelOutput = tk.Label(frame, textvariable=textOutput)
    labelResult = tk.Label(frame, textvariable=textResult)
    labelModelDetail = tk.Label(frame, textvariable=textModelDetail)
    labelSlider = tk.Label(frame, textvariable=textSlider)

    buttonRun = tk.Button(frame, textvariable=textRun, command=train_button_clicked, width=14)
    buttonStop = tk.Button(frame, textvariable=textStop, command=test_button_clicked, width=14)
    buttonInputFile = tk.Button(frame, textvariable=textInputFile, command=train_file_dialog, width=14)
    buttonOutputFile = tk.Button(frame, textvariable=textOutputFile, command=test_file_dialog, width=14)

    canvas = tk.Canvas(frame, bg="black", width=320, height=320)
    canvas.place(x=0, y=0)

    load_detail(test_list[0])

    # 各種ウィジェットの設置
    labelInput.grid(row=0, column=0, sticky=tk.NW, rowspan=1)
    ListboxInput.grid(row=1, column=0, sticky=tk.NW, rowspan=4)
    buttonInputFile.grid(row=6, column=0, sticky=tk.NW)

    labelOutput.grid(row=0, column=1, sticky=tk.NW)
    ListboxOutput.grid(row=1, column=1, sticky=tk.NW, rowspan=4)
    buttonOutputFile.grid(row=6, column=1, sticky=tk.NW)

    labelResult.grid(row=0, column=2, sticky=tk.NW)
    ListboxResult.grid(row=1, column=2, sticky=tk.NW, rowspan=4)

    labelModelDetail.grid(row=0, column=3, sticky=tk.NW)
    canvas.grid(row=1, column=3, sticky=tk.NW, rowspan=4)

    buttonRun.grid(row=6, column=3, sticky=tk.NW)
    buttonStop.grid(row=7, column=3, sticky=tk.NW)

    labelSlider.grid(row=8, column=3, sticky=tk.NW)

    # スライダーの作成
    var_scale = tk.DoubleVar()
    var_scale.set(slider_index)
    scale = tk.Scale(
        frame,
        variable=var_scale,
        orient=tk.HORIZONTAL,
        tickinterval=20,
        length=200,
    )
    scale.grid(row=9, column=3, sticky=tk.NW)
    print(var_scale.get())
    scale.bind("<ButtonRelease-1>", slider_changed)

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=20, pady=10)

    root.mainloop()

if __name__ == '__main__':
    main()



