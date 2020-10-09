import os
import cv2
import numpy

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

#マウスの操作があるとき呼ばれる関数
def callback(event, x, y, flags, param):
    global pt, m, n, p
    print(x,y)

    #マウスの左ボタンがクリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN:
        m = n
        pt[n] = (x, y)
        #print(n)
        #print(pt)
        n = n + 1
        p = p + 1

    #マウスの右ボタンがクリックされたとき
    if event == cv2.EVENT_RBUTTONDOWN and n > 0:
        #print(m, n)
        #print(pt[m])
        pt.pop(m)
        m = m - 1
        n = n - 1
        p = p + 1

def display_ui(img,model_list):
    x = 2
    y = 2
    w = 400
    h = 20
    margin = 2

    for model in model_list:
        print(model)

        cv2.rectangle(img, (x, y), (x + w, y + h), (128,128,128), thickness=-1)

        text_position = (x+4, y-8)

        color = (0,0,0)
        #hsv_to_rgb(256 * obj.category / len(category), 255, 255)
        fontScale = 0.5

        cv2.putText(
            img,
            model["script"],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y+h+margin

# ui
model_list = search_model()
img = numpy.zeros((480,640,3)).astype(numpy.uint8)
display_ui(img,model_list)

cv2.imshow('frame', img)
cv2.setMouseCallback("frame", callback)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame', img)

