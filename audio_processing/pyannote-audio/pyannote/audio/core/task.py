# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from typing import Dict, List, Literal, Optional, Sequence, Text, Tuple, Union

import scipy.special

from pyannote.database.protocol.protocol import Scope, Subset


Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)


# Type of machine learning problem
class Problem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4
    # any other we could think of?


# A task takes an audio chunk as input and returns
# either a temporal sequence of predictions
# or just one prediction for the whole audio chunk
class Resolution(Enum):
    FRAME = 1  # model outputs a sequence of frames
    CHUNK = 2  # model outputs just one vector for the whole chunk


class UnknownSpecificationsError(Exception):
    pass


@dataclass
class Specifications:
    problem: Problem
    resolution: Resolution

    # (maximum) chunk duration in seconds
    # duration: float
    duration: float = 10.0

    # (for variable-duration tasks only) minimum chunk duration in seconds
    min_duration: Optional[float] = None

    # use that many seconds on the left- and rightmost parts of each chunk
    # to warm up the model. This is mostly useful for segmentation tasks.
    # While the model does process those left- and right-most parts, only
    # the remaining central part of each chunk is used for computing the
    # loss during training, and for aggregating scores during inference.
    # Defaults to 0. (i.e. no warm-up).
    warm_up: Optional[Tuple[float, float]] = (0.0, 0.0)

    # (for classification tasks only) list of classes
    classes: Optional[List[Text]] = None
    # classes: Optional[List[Text]] = ['speaker#1', 'speaker#2', 'speaker#3']

    # (for powerset only) max number of simultaneous classes
    # (n choose k with k <= powerset_max_classes)
    # powerset_max_classes: Optional[int] = None
    powerset_max_classes: Optional[int] = 2

    # whether classes are permutation-invariant (e.g. diarization)
    # permutation_invariant: bool = False
    permutation_invariant: bool = True


    @cached_property
    def powerset(self) -> bool:
        if self.powerset_max_classes is None:
            return False

        if self.problem != Problem.MONO_LABEL_CLASSIFICATION:
            raise ValueError(
                "`powerset_max_classes` only makes sense with multi-class classification problems."
            )

        return True

    @cached_property
    def num_powerset_classes(self) -> int:
        # compute number of subsets of size at most "powerset_max_classes"
        # e.g. with len(classes) = 3 and powerset_max_classes = 2:
        # {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}
        return int(
            sum(
                scipy.special.binom(len(self.classes), i)
                for i in range(0, self.powerset_max_classes + 1)
            )
        )

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

