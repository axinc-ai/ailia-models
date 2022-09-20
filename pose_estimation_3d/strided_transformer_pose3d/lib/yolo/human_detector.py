import os

from lib.yolo.util import *
from lib.yolo import preprocess


def yolo_human_det(img, model=None, reso=416, confidence=0.8):
    inp_dim = reso
    if type(img) == str:
        assert os.path.isfile(img), 'The image path does not exist'
        img = cv2.imread(img)

    img, ori_img, img_dim, ratio = preprocess.preproc(img, (inp_dim,inp_dim))
    output = model.run(img.astype('float32')[None, :, :, :])[0]
    output = postprocess(output, (416, 416))[0]
    bboxs, scores = predictions_to_object(output, ratio, confidence, confidence)

    return np.array(bboxs), np.array(scores)
