import numpy as np

class RecurrentFlowCompleteNet():

    def __init__(self, net, is_onnx=False):
        self.net = net
        self.is_onnx = is_onnx

    def forward_bidirect_flow(self, masked_flows_bi, masks):
        """
        Args:
            masked_flows_bi: [masked_flows_f, masked_flows_b] | (b t-1 2 h w), (b t-1 2 h w)
            masks: b t 1 h w
        """
        masks_forward = masks[:, :-1, ...]#.contiguous()
        masks_backward = masks[:, 1:, ...]#.contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bi[0] * (1-masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1-masks_backward)

        # -- completion --
        # forward
        if self.is_onnx:
            pred_flows_forward = self.net.run(None, {self.net.get_inputs()[0].name: masked_flows_forward,
                                                     self.net.get_inputs()[1].name: masks_forward})[0]
        else:
            pred_flows_forward = self.net.run([masked_flows_forward, masks_forward])[0]

        # backward
        masked_flows_backward = np.flip(masked_flows_backward, axis=[1])
        masks_backward = np.flip(masks_backward, axis=[1])
        if self.is_onnx:
            pred_flows_backward = self.net.run(None, {self.net.get_inputs()[0].name: masked_flows_backward,
                                                     self.net.get_inputs()[1].name: masks_backward})[0]
        else:
            pred_flows_backward = self.net.run([masked_flows_backward, masks_backward])[0]

        pred_flows_backward = np.flip(pred_flows_backward, axis=[1])

        return pred_flows_forward, pred_flows_backward


    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...]#.contiguous()
        masks_backward = masks[:, 1:, ...]#.contiguous()

        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1-masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1-masks_backward)

        return pred_flows_forward, pred_flows_backward


class InpaintGenerator:

    def __init__(self, img_prop_module, net, is_onnx=False):
        self.img_prop_module = img_prop_module
        self.net = net
        self.is_onnx = is_onnx

    def img_propagation(self, masked_frames, completed_flows, masks):
        # masked_frames has size [1, 80, 3, 240, 432]
        # completed_flows[0] has size [1, 79, 2, 240, 432]
        # completed_flows[1] has size [1, 79, 2, 240, 432]
        # masks has size [1, 80, 1, 240, 432]
        if self.is_onnx:
            _, _, prop_frames, updated_masks = self.img_prop_module.run(None, {self.img_prop_module.get_inputs()[0].name: masked_frames,
                                                                               self.img_prop_module.get_inputs()[1].name: completed_flows[0],
                                                                               self.img_prop_module.get_inputs()[2].name: completed_flows[1],
                                                                               self.img_prop_module.get_inputs()[3].name: masks})
        else:
            _, _, prop_frames, updated_masks = self.img_prop_module.run(masked_frames, completed_flows[0], completed_flows[1], masks)

        return prop_frames, updated_masks

    def __call__(self, masked_frames, completed_flows, masks_in, masks_updated, num_local_frames):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """
        if self.is_onnx:
            pred_img = self.net.run(None, {self.img_prop_module.get_inputs()[0].name: masked_frames,
                                           self.img_prop_module.get_inputs()[1].name: completed_flows,
                                           self.img_prop_module.get_inputs()[2].name: masks_in,
                                           self.img_prop_module.get_inputs()[3].name: masks_updated,
                                           self.img_prop_module.get_inputs()[4].name: num_local_frames})[0]
        else:
            pred_img = self.net.run(masked_frames, completed_flows, masks_in, masks_updated, num_local_frames)[0]

        return pred_img

