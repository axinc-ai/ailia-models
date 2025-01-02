import numpy as np
import cv2

class GazeLLE():
    def __init__(self, backbone, decoder, inout=False, out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.inout = inout
        self.out_size = out_size

    @staticmethod
    def repeat_array(tensor, repeat_counts):
        indices = np.repeat(np.arange(len(tensor)), repeat_counts)
        return tensor[indices]
    
    @staticmethod
    def get_input_head_maps(bboxes, featmap_h=32, featmap_w=32):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:  # Create an empty map if bounding boxes are not provided
                    img_head_maps.append(np.zeros((featmap_h, featmap_w), dtype=np.float32))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    xmin = round(xmin * featmap_w)
                    ymin = round(ymin * featmap_h)
                    xmax = round(xmax * featmap_w)
                    ymax = round(ymax * featmap_h)
                    head_map = np.zeros((featmap_h, featmap_w), dtype=np.float32)
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(np.stack(img_head_maps))
        return head_maps
    
    @staticmethod
    def split_array(tensor, split_counts):
        indices = np.cumsum([0] + split_counts)
        return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]
    
    def resize_array(self, tensor):
        tensor = np.transpose(tensor, (1, 2, 0))
        tensor = cv2.resize(tensor, self.out_size, interpolation=cv2.INTER_LINEAR)
        if len(tensor.shape) == 2:
            tensor = tensor[:, :, np.newaxis]
        return np.transpose(tensor, (2, 0, 1))

    def run(self, images, bboxes):
        num_ppl_per_img = [len(bbox_list) for bbox_list in bboxes]

        # Backbone inference
        backbone_features = self.backbone.predict({"input": images})[0]

        backbone_features = self.repeat_array(backbone_features, num_ppl_per_img)
        head_maps = np.concatenate(self.get_input_head_maps(bboxes), axis=0)

        # Decoder inference
        decoder_outputs = self.decoder.predict({"input": backbone_features, "head_maps": head_maps})
        
        resized_heatmaps = self.resize_array(decoder_outputs[0])
        heatmap_preds = self.split_array(resized_heatmaps, num_ppl_per_img)
        inout_preds = self.split_array(decoder_outputs[1], num_ppl_per_img) if self.inout else None

        return {"heatmap": heatmap_preds, "inout": inout_preds}
