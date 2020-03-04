import os

import numpy as np
from openvino.inference_engine import IENetwork, IECore


class InferenceEngineOpenVINO:
    def __init__(self, net_model_xml_path, device):
        self.device = device

        net_model_bin_path = os.path.splitext(net_model_xml_path)[0] + '.bin'
        self.net = IENetwork(model=net_model_xml_path, weights=net_model_bin_path)
        required_input_key = {'data'}
        assert required_input_key == set(self.net.inputs.keys()), \
            'Demo supports only topologies with the following input key: {}'.format(', '.join(required_input_key))
        required_output_keys = {'features', 'heatmaps', 'pafs'}
        assert required_output_keys.issubset(self.net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        self.ie = IECore()
        self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=device)

    def infer(self, img):
        input_layer = next(iter(self.net.inputs))
        n, c, h, w = self.net.inputs[input_layer].shape
        if h != img.shape[0] or w != img.shape[1]:
            self.net.reshape({input_layer: (n, c, img.shape[0], img.shape[1])})
            self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=self.device)
        img = np.transpose(img, (2, 0, 1))[None, ]

        inference_result = self.exec_net.infer(inputs={'data': img})

        inference_result = (inference_result['features'][0],
                            inference_result['heatmaps'][0], inference_result['pafs'][0])
        return inference_result

