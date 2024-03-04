# MIT License
#
# Copyright (c) 2023- CNRS
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


import torch

from pyannote.audio.utils.powerset import Powerset


def test_roundtrip():
    for num_classes in range(2, 5):
        for max_set_size in range(1, num_classes + 1):
            powerset = Powerset(num_classes, max_set_size)

            # simulate a sequence where each frame is assigned to a different powerset class
            one_sequence = [
                [0] * powerset.num_powerset_classes
                for _ in range(powerset.num_powerset_classes)
            ]
            for i in range(powerset.num_powerset_classes):
                one_sequence[i][i] = 1.0

            # make a batch out of this sequence and the same sequence in reverse order
            batch_powerset = torch.tensor([one_sequence, one_sequence[::-1]])

            # convert from powerset to multi-label
            batch_multilabel = powerset.to_multilabel(batch_powerset)

            # convert batch back to powerset
            reconstruction = powerset.to_powerset(batch_multilabel)

            assert torch.equal(batch_powerset, reconstruction)


def test_permutate_powerset():
    for num_classes in range(1, 6):
        for max_set_size in range(1, num_classes + 1):
            powerset = Powerset(num_classes, max_set_size)

            # create (num_powerset_class, num_powerset_class)-shaped tensor, where each frame is assigned to a different powerset class
            # and convert it to its multi-label equivalent
            t1 = torch.nn.functional.one_hot(
                torch.arange(powerset.num_powerset_classes),
                powerset.num_powerset_classes,
            )
            t1_ml = powerset.to_multilabel(t1)

            # then permutate the powerset class in powerset space AND the multilabel equivalent in its native space
            # and check it has the same result.
            # perm = torch.randperm(num_classes)
            perm = tuple(torch.randperm(num_classes).tolist())
            t1_ml_perm = t1_ml[:, perm]
            perm_ps = powerset.permutation_mapping[perm]
            t1_ps_perm = t1[..., perm_ps]
            t1_ps_perm_ml = powerset.to_multilabel(t1_ps_perm)

            assert t1_ml_perm.equal(t1_ps_perm_ml)
