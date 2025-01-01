import cv2
import numpy as np

from scipy.ndimage import zoom

from util import *

class BasePredictor:
    def __init__(self, model):
        self.imgsz = None
        self.model = model
        
    def pre_transform(self, im):
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = False
        stride = 32
        return [LetterBox(self.imgsz, auto=auto, stride=stride)(image=x) for x in im]

    def setup_source(self, source):
        """Sets up source and inference mode."""
        stride = 32
        imgsz = 1024
        self.imgsz = check_imgsz(imgsz, stride=stride, min_dim=2)  # check image size

        source = np.asarray(source)[:, :, ::-1]
        source = cv2.cvtColor(source , cv2.COLOR_BGR2RGB)
        source = np.ascontiguousarray(source)  # contiguous
        return [source]
 
    def stream_inference(self, source=None,conf=0.4,iou=0.9):
        """Streams real-time inference on camera feed and saves results to file."""

        im0s = self.setup_source(source)

        # Preprocess
        im = np.stack(self.pre_transform(im0s))
        im = im[..., ::-1].transpose((0, 3, 1, 2)) # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = np.float32(im) / 255.0

        preds = self.model.run(im)

        results = self.postprocess(preds, im, im0s,conf,iou)

        return results

class FastSAMPredictor(BasePredictor):

    def __init__(self, model):
        super().__init__(model)

    def __call__(self, source=None, conf=0.4,iou=0.9,**kwargs):
        return list(self.stream_inference(source))  # merge list of Result into one
 
    def postprocess(self, preds, img, orig_imgs,conf,iou):
        agnostic_nms = False,
        max_det = 300
        classes = None

        preds = [ np.array(preds[0]),
                    [[np.array(preds[1]),
                      np.array(preds[2]),
                      np.array(preds[3])],

                      np.array(preds[4]),
                      np.array(preds[5])]]

        """TODO: filter by classes."""
        p = non_max_suppression(preds[0],
                                    conf,
                                    iou,
                                    agnostic=agnostic_nms,
                                    max_det=max_det,
                                    classes=classes)

        results = []
        if len(p) == 0 or len(p[0]) == 0:
            print("No object detected.")
            return results

        full_box = np.zeros_like(p[0][0])
        full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3], img.shape[2], 1.0, 1.0
        full_box = full_box.reshape(1, -1)
        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:])

        if critical_iou_index.size != 0:
            full_box[0][4] = p[0][critical_iou_index][:,4]
            full_box[0][6:] = p[0][critical_iou_index][:,6:]
            p[0][critical_iou_index] = full_box

       
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported

        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = "img.jpg"
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append({"orig_img":orig_img, "path":img_path, "boxes":pred[:, :6], "masks":masks})
                continue
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            masks = process_mask_native(proto[i], pred[:, 6:],pred[:, :4], orig_img.shape[:2])  # HWC
            results.append(
                Results(orig_img=orig_img, path=img_path, names=None, boxes=pred[:, :6], masks=masks))
        return results

