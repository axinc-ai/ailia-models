# ailia MODELS launcher

import os
import cv2
import numpy
import subprocess
import shutil

WINDOW_WIDTH = 1608
WINDOW_HEIGHT = 484

BUTTON_WIDTH = 400
BUTTON_HEIGHT = 20
BUTTON_MARGIN = 2

IGNORE_LIST=["commercial_model", "validation", ".git", "log", "prnet", "bert", "illustration2vec", "etl", "vggface2", "audio_processing"]

def search_model():
    file_list=[]
    for current, subfolders, subfiles in os.walk("./"):
        file_list.append(current)

    file_list.sort()

    model_list=[]
    category_list={}
    model_exist={}
    for current in file_list:
        current = current.replace("\\","/")
        files = current.split("/")
        if len(files)==3:
            if (files[1] in IGNORE_LIST) or (files[2] in IGNORE_LIST):
                continue
            if files[2] in model_exist:
                continue
            script = "./"+files[1]+"/"+files[2]+"/"+files[2]+".py"
            if os.path.exists(script):
                if not(files[1] in category_list):
                    category_list[files[1]]=len(category_list)
                category_id=category_list[files[1]]
                model_list.append({"category":files[1],"category_id":category_id,"model":files[2]})
                model_exist[files[2]]=True
    return model_list,len(category_list)

def mouse_callback(event, x, y, flags, param):
    global mx,my,click_trig
    if event == cv2.EVENT_LBUTTONDOWN:
        click_trig = True
    mx = x
    my = y

mx = 0
my = 0
click_trig = False

def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        numpy.array([[[h, s, v]]], dtype=numpy.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def display_ui(img,model_list,category_cnt):
    global mx,my,click_trig

    x = BUTTON_MARGIN
    y = BUTTON_MARGIN
    w = BUTTON_WIDTH
    h = BUTTON_HEIGHT

    for model in model_list:
        color = hsv_to_rgb(256 * model["category_id"] / (category_cnt+1), 128, 255)

        if mx >= x and mx <= x+w and my >= y and my <= y+h:
           color = (255,255,255)
           if click_trig:
                dir="./"+model["category"]+"/"+model["model"]+"/"
                cmd="python"
                if shutil.which("python3"):
                    cmd="python3"
                subprocess.run([cmd,model["model"]+".py","-v 0"],cwd=dir)
                click_trig = False

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)

        text_position = (x+4, y+int(BUTTON_HEIGHT/2)+4)

        color = (0,0,0)
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

        y=y + h + BUTTON_MARGIN

        if y>=WINDOW_HEIGHT:
            y = BUTTON_MARGIN
            x = x + w + BUTTON_MARGIN
    
    click_trig = False

def main():
    model_list,category_cnt = search_model()
    img = numpy.zeros((WINDOW_HEIGHT,WINDOW_WIDTH,3)).astype(numpy.uint8)

    cv2.imshow('ailia MODELS', img)
    cv2.setMouseCallback("ailia MODELS", mouse_callback)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        display_ui(img,model_list,category_cnt)
        cv2.imshow('ailia MODELS', img)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
