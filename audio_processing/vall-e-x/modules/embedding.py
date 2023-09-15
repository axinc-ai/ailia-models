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

import math

import torch
import torch.nn as nn

import ailia
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = self.net.run([x.numpy()])[0]
        y = torch.from_numpy(y)
        return y

    def load_onnx(self, net):
        self.net = net

class TokenEmbeddingLayers(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, layer_id):
        layer_id = np.array(layer_id, dtype=np.int64) # constant type (shape = ())
        y = self.net.run([x.numpy(), layer_id])[0]
        y = torch.from_numpy(y)
        return y
    
    def load_onnx(self, net):
        self.net = net

class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
        alpha_parameter: float =1.0
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_parameter]), requires_grad=alpha)

    def forward(self, x: torch.Tensor):
        y = self.net.run([x.numpy(), self.alpha.numpy()])[0]
        y = torch.from_numpy(y)
        return y
    
    def load_onnx(self, net):
        self.net = net
