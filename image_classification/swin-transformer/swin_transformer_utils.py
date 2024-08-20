import os
import sys
import numpy as np
import cv2

MAX_CLASS_COUNT = 3

RECT_WIDTH = 640
RECT_HEIGHT = 20
RECT_MARGIN = 2

def get_top_scores(output):
    topk = MAX_CLASS_COUNT
    unsorted_max_indices = np.argpartition(-output, topk)[:topk]
    score = output[unsorted_max_indices]
    indices = np.argsort(-score)
    max_k_indices = unsorted_max_indices[indices]
    return max_k_indices, indices


def print_results(output, labels):
    max_k_indices, indices = get_top_scores(output)

    print('==============================================================')
    print(f'class_count={MAX_CLASS_COUNT}')
    for idx in range(MAX_CLASS_COUNT):
        print(f'+ idx={idx}')
        print(f'  category={max_k_indices[idx]}['
              f'{labels[max_k_indices[idx]][1]} ]')
        print(f'  prob={output[max_k_indices[idx]]}')


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def plot_results(input_image, output, labels, logging=True):
    x = RECT_MARGIN
    y = RECT_MARGIN
    w = RECT_WIDTH
    h = RECT_HEIGHT

    max_k_indices, indices = get_top_scores(output)

    if logging:
        print('==============================================================')
        print(f'class_count={MAX_CLASS_COUNT}')
    for idx in range(MAX_CLASS_COUNT):
        if logging:
            print(f'+ idx={idx}')
            print(f'  category={max_k_indices[idx]}['
                f'{labels[max_k_indices[idx]][1]} ]')
            print(f'  prob={output[max_k_indices[idx]]}')

        text = f'category={max_k_indices[idx]}[{labels[max_k_indices[idx]][1]} ] prob={output[max_k_indices[idx]]}'

        color = hsv_to_rgb(256 * max_k_indices[idx] / (len(labels)+1), 128, 255)

        cv2.rectangle(input_image, (x, y), (x + w, y + h), color, thickness=-1)
        text_position = (x+4, y+int(RECT_HEIGHT/2)+4)

        color = (0,0,0)
        fontScale = 0.5

        cv2.putText(
            input_image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y + h + RECT_MARGIN
