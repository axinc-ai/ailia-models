import os
import sys
import numpy as np
import cv2


def preprocessing_img(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        print(f'[ERROR] {image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def plot_results(detector, img, category, logging=True):
    h, w = img.shape[0], img.shape[1]
    count = detector.get_object_count()
    if logging:
        print(f'object_count={count}')
    
    for idx in range(count):
        obj = detector.get_object(idx)
        # print result
        if logging:
            print(f'+ idx={idx}')        
            print(
                f'  category={obj.category}[ {category[obj.category]} ]'
            )
            print(f'  prob={obj.prob}')
            print(f'  x={obj.x}')
            print(f'  y={obj.y}')
            print(f'  w={obj.w}')
            print(f'  h={obj.h}')
        top_left = (int(w*obj.x), int(h*obj.y))
        bottom_right = (int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)))
        text_position = (int(w*obj.x)+4, int(h*(obj.y+obj.h)-8))

        # update image
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255, 255), 4)

        color = hsv_to_rgb(255 * obj.category / 80, 255, 255)
        fontScale = w / 512.0
        cv2.rectangle(img, top_left, bottom_right, color, int(fontScale))

        cv2.putText(
            img,
            category[obj.category],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )
    return img


