# ailia MODELS launcher

import os
import cv2
import numpy
import subprocess
import shutil

mx = 0
my = 0
click_trig = False

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

IGNORE_LIST=["commercial_model","validation",".git","log"]

def search_model():
    model_list=[]
    category_list={}
    model_exist={}
    for current, subfolders, subfiles in os.walk("./"):
        files = current.split("/")
        if len(files)==3:
            if (files[1] in IGNORE_LIST) or (files[2] in IGNORE_LIST):
                continue
            if files[2] in model_exist:
                continue
            script = "./"+files[1]+"/"+files[2]+"/"+files[2]+".py"
            #print(script)
            if os.path.exists(script):
                if not(files[1] in category_list):
                    category_list[files[1]]=len(category_list)
                category_id=category_list[files[1]]
                model_list.append({"category":files[1],"category_id":category_id,"model":files[2],"script":script})
                model_exist[files[2]]=True
    return model_list,len(category_list)

def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        numpy.array([[[h, s, v]]], dtype=numpy.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def callback(event, x, y, flags, param):
    global mx,my,click_trig
    if event == cv2.EVENT_LBUTTONDOWN:
        click_trig = True
    mx = x
    my = y

def display_ui(img,model_list,category_cnt):
    global mx,my,click_trig

    x = 2
    y = 2
    w = 200
    h = 20
    margin = 2

    for model in model_list:
        color = hsv_to_rgb(256 * model["category_id"] / (category_cnt+1), 128, 255)

        if mx >= x and mx <= x+w and my >= y and my <= y+h:
           color = (192,192,192)
           if click_trig:
                print(model["script"])
                dir="./"+model["category"]+"/"+model["model"]+"/"
                cmd="python"
                if shutil.which("python3"):
                    cmd="python3"
                subprocess.run([cmd,model["model"]+".py","-v 0"],cwd=dir)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)

        text_position = (x+4, y+12)

        color = (0,0,0)
        fontScale = 0.5

        cv2.putText(
            img,
            model["model"],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y+h+margin

        if y>=WINDOW_HEIGHT:
            y = 2
            x = x + w + 2
    
    click_trig = False

# ui
model_list,category_cnt = search_model()
img = numpy.zeros((WINDOW_HEIGHT,WINDOW_WIDTH,3)).astype(numpy.uint8)

cv2.imshow('ailia MODELS', img)
cv2.setMouseCallback("ailia MODELS", callback)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    display_ui(img,model_list,category_cnt)
    cv2.imshow('ailia MODELS', img)

cv2.destroyAllWindows()