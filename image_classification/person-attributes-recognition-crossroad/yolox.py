import sys
import cv2

from yolox_utils import  postprocess, predictions_to_object
from yolox_utils import preproc as preprocess

# import original modules
sys.path.append('../../util')

from detector_utils import reverse_letterbox
                         
# ======================
# Parameters
# ======================
MODEL_PARAMS = {'yolox_s': {'input_shape': [640, 640]},}


HEIGHT = MODEL_PARAMS['yolox_s']['input_shape'][0]
WIDTH  = MODEL_PARAMS['yolox_s']['input_shape'][1]

# ======================
# Main functions
# ======================
def yolox(raw_img,detector,nms_iou,score_th):
    # input image loop
    img, ratio = preprocess(raw_img, (HEIGHT, WIDTH))


    output = detector.run(img[None, :, :, :])
    predictions = postprocess(output[0], (HEIGHT, WIDTH))[0]
    detect_object = predictions_to_object(predictions, raw_img, ratio, nms_iou, score_th)
    detect_object = reverse_letterbox(detect_object, raw_img, (raw_img.shape[0], raw_img.shape[1]))

    person = None
    for i in range(len(detect_object)):
        if not detect_object[0][0] == 0:
            continue
        person = True
        img_h,img_w,_ = raw_img.shape
        x = int(detect_object[i][2] * img_w)
        y = int(detect_object[i][3] * img_h)
        w = int(detect_object[i][4] * img_w)
        h = int(detect_object[i][5] * img_h)
    
    if not person is None:
        return raw_img[y:y+h,x:x+w,:]
    else:
        return raw_img

