import numpy as np
from scipy.special import softmax
from max_deeplab_utils.visualize import display_instances, roll_image
from matplotlib import pyplot as plt

category_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
                 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
                 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
                 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
                 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
                 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
                 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 92: 'banner', 93: 'blanket',
                 95: 'bridge', 100: 'cardboard', 107: 'counter', 109: 'curtain', 112: 'door-stuff', 118: 'floor-wood', 119: 'flower',
                 122: 'fruit', 125: 'gravel', 128: 'house', 130: 'light', 133: 'mirror-stuff', 138: 'net', 141: 'pillow', 144: 'platform',
                 145: 'playingfield', 147: 'railroad', 148: 'river', 149: 'road', 151: 'roof', 154: 'sand', 155: 'sea', 156: 'shelf',
                 159: 'snow', 161: 'stairs', 166: 'tent', 168: 'towel', 171: 'wall-brick', 175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood',
                 178: 'water-other', 180: 'window-blind', 181: 'window-other', 184: 'tree-merged', 185: 'fence-merged', 186: 'ceiling-merged',
                 187: 'sky-other-merged', 188: 'cabinet-merged', 189: 'table-merged', 190: 'floor-other-merged', 191: 'pavement-merged',
                 192: 'mountain-merged', 193: 'grass-merged', 194: 'dirt-merged', 195: 'paper-merged', 196: 'food-other-merged',
                 197: 'building-other-merged', 198: 'rock-merged', 199: 'wall-other-merged', 200: 'rug-merged', 201: 'no_class'}

def shape_values(classes, class_confidence):
    outvalues1 = []
    outvalues2 = []
    i = 0
    for cls, confidence in zip(classes, class_confidence):
        values = (cls > 0) & (confidence > 0.7)
        out1 = [i] * sum(values)
        out2 = np.arange(len(values))[values].tolist()
        outvalues1 += out1
        outvalues2 += out2
        i += 1

    return np.array(outvalues1), np.array(outvalues2)

def shape_pred(out, N):
    instance_probs = softmax(out[0], axis=1)
    instances = instance_probs.argmax(axis=1)
    instances = np.identity(N)[instances]
    instances = np.transpose(instances, (0, 3, 1, 2))

    class_confidence = softmax(out[1], axis=-1).max(-1)
    classes = softmax(out[1], axis=-1).argmax(-1)

    semantic = softmax(out[2], axis=1).argmax(axis=1)

    # filter out low confidence instances from predictions
    keep_pred_instances = shape_values(classes, class_confidence)

    return instances, classes, keep_pred_instances

def visualize(image, instances, classes, keep_pred_instances):

    pred_instances = []
    pred_classes = []
    pred_class_names = []

    keep_pred = keep_pred_instances[1][keep_pred_instances[0] == 0]

    pred_instances.append(instances[0, keep_pred])
    pred_classes.append(classes[0, keep_pred])
    if len(pred_classes[0]) > 1:
        pred_class_names.append([category_dict[l] for l in pred_classes[0]])
    else:
        pred_class_names.append(['None'])
        
    fig, ax = plt.subplots(1, figsize=(12, 6), tight_layout=True)
    display_instances(image, pred_instances[0],pred_classes[0], pred_class_names[0], ax=ax)
    plt.tight_layout()
    return fig