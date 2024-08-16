# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract class for sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Sampler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X

        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))

        return flat_X

    @abstractmethod
    def select_batch(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)
