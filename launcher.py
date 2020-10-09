# ailia MODELS launcher

import os
import cv2
import numpy

mx = 0
my = 0
click_trig = False

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

def search_model():
    model_list=[]
    for current, subfolders, subfiles in os.walk("./"):
        if len(subfolders)==0:
            files = current.split("/")
            if len(files)==4:
                script = "./"+files[1]+"/"+files[2]+"/"+files[2]+".py"
                print(script)
                if os.path.exists(script):
                    model_list.append({"category":files[1],"model":files[2],"script":script})
    return model_list

def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def callback(event, x, y, flags, param):
    global mx,my,click_trig
    if event == cv2.EVENT_LBUTTONDOWN:
        click_trig = True
    mx = x
    my = y

def display_ui(img,model_list):
    global mx,my,click_trig

    x = 2
    y = 2
    w = 200
    h = 20
    margin = 2

    for model in model_list:
        color = (128,128,128)

        if mx >= x and mx <= x+w and my >= y and my <= y+h:
            color = (192,192,192)
            if click_trig:
                print(model["script"])

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=-1)

        text_position = (x+4, y+12)

        color = (0,0,0)
        #hsv_to_rgb(256 * obj.category / len(category), 255, 255)
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
model_list = search_model()
img = numpy.zeros((WINDOW_HEIGHT,WINDOW_WIDTH,3)).astype(numpy.uint8)

cv2.imshow('frame', img)
cv2.setMouseCallback("frame", callback)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    display_ui(img,model_list)
    cv2.imshow('frame', img)

