import numpy as np

import ailia


def box_cxcywh_to_xyxy(x):
    output = np.zeros(x.shape)
    x_c = x[:,:,0]
    y_c = x[:,:,1]
    w = x[:,:,2]
    h = x[:,:,3]

    output[:, :, 0] = (x_c - 0.5 * w)
    output[:, :, 1] = (y_c - 0.5 * h)
    output[:, :, 2] = (x_c + 0.5 * w)
    output[:, :, 3] = (y_c + 0.5 * h)
    return output

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

class PreProcess:
    def __init__(self, resize_img, mean, std):
        resize_img = resize_img.transpose(2, 0, 1)
        resize_img = np.expand_dims(resize_img, axis=0) / 255
        self.img = resize_img
        self.mean = mean
        self.std = std

    def create_masks(self):
        b, c, h, w = self.img.shape
        mask = np.zeros((b, h, w))
        for im, ma in zip(self.img, mask):
            ma[: im.shape[1], :im.shape[2]] = 1
        return mask == 0

    def normalization(self):
        img = self.img
        mean = self.mean
        std = self.std
        for im in img:
            for i in range(im.shape[0]):
                im[i] = (im[i] - mean[i]) / std[i]

        img = np.array(img, dtype='float32')
        return img

class CocoClassMapper():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.origin2compact_mapper = {int(k):v-1 for k,v in self.category_map_str.items()}
        self.compact2origin_mapper = {int(v-1):int(k) for k,v in self.category_map_str.items()}

    def origin2compact(self, idx):
        return self.origin2compact_mapper[int(idx)]

    def compact2origin(self, idx):
        return self.compact2origin_mapper[int(idx)]

class PostProcess:
    def __init__(self, num_select=300):
        self.num_select = num_select

    def forward(self, outputs):
        cocomap = CocoClassMapper()

        num_select = self.num_select
        out_logits = outputs[0]
        out_bbox = outputs[1]

        prob = sigmoid(out_logits)
        prob_flatten = prob.reshape(out_logits.shape[0], -1)
        topk_indexes = (-prob_flatten).argsort()[:, :num_select]
        topk_values = prob_flatten[:,topk_indexes][0]

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        labels = [[cocomap.origin2compact(str(int(l))) for l in labels[0]]]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes[:,topk_boxes[0],:]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class Detect:
    def __init__(self, session, img, args, thershold=0.4):
        self.session = session
        self.img = img
        if args.onnx:
            self.input_name1 = session.get_inputs()[0].name
            self.input_name2 = session.get_inputs()[1].name
            self.output_name1 = session.get_outputs()[0].name
            self.output_name2 = session.get_outputs()[1].name
        self.thershold = thershold


    def detect(self, args):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        preprosess = PreProcess(self.img, mean, std)
        img = preprosess.normalization()
        mask = preprosess.create_masks()

        if args.onnx:
            out_logits = self.session.run([self.output_name1], {self.input_name1: img, self.input_name2: mask})[0]
            out_bbox = self.session.run([self.output_name2], {self.input_name1: img, self.input_name2: mask})[0]
        else:
            outputs = self.session.run({self.input_name1: img, self.input_name2: mask})
            out_logits = outputs[0]
            out_bbox = outpus[1]

        postprocessors = PostProcess()
        results = postprocessors.forward((out_logits, out_bbox))[0]

        terms = results['scores'] > self.thershold
        confs = results['scores'][terms]
        bboxs = results['boxes'][terms]
        labels = np.array(results['labels'])[terms]

        output = []
        for i, box in enumerate(bboxs):
            x1, y1, x2, y2 = box
            c = int(labels[i])
            r = ailia.DetectorObject(
                category=c,
                prob=confs[i],
                x=x1 ,
                y=y1 ,
                w=(x2 - x1) ,
                h=(y2 - y1),
            )
            output.append(r)

        return output