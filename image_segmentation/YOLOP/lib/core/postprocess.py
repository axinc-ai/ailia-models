import torch
from lib.utils import is_parallel
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
from sklearn.cluster import DBSCAN


def build_targets(cfg, predictions, targets, model):
    '''
    predictions
    [16, 3, 32, 32, 85]
    [16, 3, 16, 16, 85]
    [16, 3, 8, 8, 85]
    torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
    [32,32,32,32]
    [16,16,16,16]
    [8,8,8,8]
    targets[3,x,7]
    t [index, class, x, y, w, h, head_index]
    '''
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[model.module.detector_index] if is_parallel(model) \
        else model.model[model.detector_index]  # Detect() module
    # print(type(model))
    # det = model.model[model.detector_index]
    # print(type(det))
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    
    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    
    for i in range(det.nl):
        anchors = det.anchors[i] #[3,2]
        gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        # Match targets to anchors
        t = targets * gain

        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < cfg.TRAIN.ANCHOR_THRESHOLD  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

def morphological_process(image, kernel_size=5, func_type=cv2.MORPH_CLOSE):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, func_type, kernel, iterations=1)

    return closing

def connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # print(gray_image.dtype)
    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

def if_y(samples_x):
    for sample_x in samples_x:
        if len(sample_x):
            # if len(sample_x) != (sample_x[-1] - sample_x[0] + 1) or sample_x[-1] == sample_x[0]:
            if sample_x[-1] == sample_x[0]:
                return False
    return True
    
def fitlane(mask, sel_labels, labels, stats):
    H, W = mask.shape
    for label_group in sel_labels:
        states = [stats[k] for k in label_group]
        x, y, w, h, _ = states[0]
        # if len(label_group) > 1:
        #     print('in')
        #     for m in range(len(label_group)-1):
        #         labels[labels == label_group[m+1]] = label_group[0]
        t = label_group[0]
        # samples_y = np.linspace(y, H-1, 30)
        # else:
        samples_y = np.linspace(y, y+h-1, 30)
        
        samples_x = [np.where(labels[int(sample_y)]==t)[0] for sample_y in samples_y]

        if if_y(samples_x):
            samples_x = [int(np.mean(sample_x)) if len(sample_x) else -1 for sample_x in samples_x]
            samples_x = np.array(samples_x)
            samples_y = np.array(samples_y)
            samples_y = samples_y[samples_x != -1]
            samples_x = samples_x[samples_x != -1]
            func = np.polyfit(samples_y, samples_x, 2)
            x_limits = np.polyval(func, H-1)
            # if (y_max + h - 1) >= 720:
            if x_limits < 0 or x_limits > W:
            # if (y_max + h - 1) > 720:
                # draw_y = np.linspace(y, 720-1, 720-y)
                draw_y = np.linspace(y, y+h-1, h)
            else:
                # draw_y = np.linspace(y, y+h-1, y+h-y)
                draw_y = np.linspace(y, H-1, H-y)
            draw_x = np.polyval(func, draw_y)
            # draw_y = draw_y[draw_x < W]
            # draw_x = draw_x[draw_x < W]
            draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
            cv2.polylines(mask, [draw_points], False, 1, thickness=15)
        else:
            # if ( + w - 1) >= 1280:
            samples_x = np.linspace(x, W-1, 30)
            # else:
            #     samples_x = np.linspace(x, x_max+w-1, 30)
            samples_y = [np.where(labels[:, int(sample_x)]==t)[0] for sample_x in samples_x]
            samples_y = [int(np.mean(sample_y)) if len(sample_y) else -1 for sample_y in samples_y]
            samples_x = np.array(samples_x)
            samples_y = np.array(samples_y)
            samples_x = samples_x[samples_y != -1]
            samples_y = samples_y[samples_y != -1]
            try:
                func = np.polyfit(samples_x, samples_y, 2)
            except:
                pass
            # y_limits = np.polyval(func, 0)
            # if y_limits > 720 or y_limits < 0:
            # if (x + w - 1) >= 1280:
            #     draw_x = np.linspace(x, 1280-1, 1280-x)
            # else:
            y_limits = np.polyval(func, 0)
            if y_limits >= H or y_limits < 0:
                draw_x = np.linspace(x, x+w-1, w+x-x)
            else:
                y_limits = np.polyval(func, W-1)
                if y_limits >= H or y_limits < 0:
                    draw_x = np.linspace(x, x+w-1, w+x-x)
                # if x+w-1 < 640:
                #     draw_x = np.linspace(0, x+w-1, w+x-x)
                else:
                    draw_x = np.linspace(x, W-1, W-x)
            draw_y = np.polyval(func, draw_x)
            draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
            cv2.polylines(mask, [draw_points], False, 1, thickness=15)
    return mask

def connect_lane(image, shadow_height=0):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    if shadow_height:
        image[:shadow_height] = 0
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)
    # ratios = []
    selected_label = []
    
    for t in range(1, num_labels, 1):
        _, _, _, _, area = stats[t]
        if area > 400:
            selected_label.append(t)
    if len(selected_label) == 0:
        return mask
    else:
        split_labels = [[label,] for label in selected_label]
        mask_post = fitlane(mask, split_labels, labels, stats)
        return mask_post








