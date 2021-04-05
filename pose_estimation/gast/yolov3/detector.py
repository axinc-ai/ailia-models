from __future__ import division
import os

import cv2

# logger
from logging import getLogger  # noqa: E402

from .util import *
from .darknet import Darknet
from . import preprocess

logger = getLogger(__name__)

this_dir = os.path.dirname(os.path.realpath(__file__))


def load_model(CUDA=None, inp_dim=416):
    cfg_file = this_dir + '/cfg/yolov3.cfg'
    weight_file = this_dir + '/checkpoint/yolov3.weights'

    if CUDA is None:
        CUDA = torch.cuda.is_available()

    # Set up the neural network
    logger.info("Loading YOLOv3 network.....")
    model = Darknet(cfg_file)
    model.load_weights(weight_file)
    logger.info("YOLOv3 network successfully loaded")

    model.net_info["height"] = inp_dim
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    return model


def yolo_human_det(img, model, reso=416, confidence=0.70):
    nms_thresh = 0.4
    inp_dim = reso
    num_classes = 80

    CUDA = torch.cuda.is_available()

    if type(img) == str:
        assert os.path.isfile(img), 'The image path does not exist'
        img = cv2.imread(img)

    img, ori_img, img_dim = preprocess.prep_image(img, inp_dim)
    img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

    with torch.no_grad():
        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()
        output = model(img, CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh, det_hm=True)

        if len(output) == 0:
            return None, None

        img_dim = img_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

    bboxs = []
    scores = []
    for i in range(len(output)):
        item = output[i]
        bbox = item[1:5].cpu().numpy()
        # conver float32 to .2f data
        bbox = [round(i, 2) for i in list(bbox)]
        score = item[5].cpu().numpy()
        bboxs.append(bbox)
        scores.append(score)
    scores = np.expand_dims(np.array(scores), 1)
    bboxs = np.array(bboxs)

    return bboxs, scores
