# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ailia
import numpy as np

class TokenEmbedding():
    def __init__(
        self,
        ort = False
    ):
        self.ort = ort

    def forward(self, x):
        if self.ort:
            y = self.net.run(None, {"x":x})[0]
        else:
            y = self.net.run([x])[0]
        return y

    def load_onnx(self, net):
        self.net = net

class TokenEmbeddingLayers():
    def __init__(
        self,
        ort = False
    ):
        self.ort = ort

    def forward(self, x, layer_id):
        layer_id = np.array(layer_id, dtype=np.int64) # constant type (shape = ())
        if self.ort:
            y = self.net.run(None, {"x":x, "onnx::Gather_1":layer_id})[0]
        else:
            y = self.net.run([x, layer_id])[0]
        return y
    
    def load_onnx(self, net):
        self.net = net

class SinePositionalEmbedding():
    def __init__(
        self,
        ort = False,
        alpha_parameter: float = 1.0
    ):
        self.ort = ort
        self.alpha = np.array([alpha_parameter], dtype=np.float32)

    def forward(self, x):
        if self.ort:
            y = self.net.run(None, {"x":x, "alpha":self.alpha})[0]
        else:
            y = self.net.run([x, self.alpha])[0]
        return y
    
    def load_onnx(self, net):
        self.net = net
