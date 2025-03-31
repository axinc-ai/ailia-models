# MIT License
#
# Copyright (c) 2022- CNRS
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

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
from pyannote_audio_utils.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote_audio_utils.core.utils.types import Label
# from pyannote_audio_utils.metrics.diarization import DiarizationErrorRate # 必要ないのでコメントアウト

from pyannote_audio_utils.audio.core.inference import Inference
from pyannote_audio_utils.audio.utils.signal import Binarize



# デバック
## ========================================================================
import csv
def write_array_to_csv(filename, data):

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
## ========================================================================



# TODO: move to dedicated module
class SpeakerDiarizationMixin:
    """Defines a bunch of methods common to speaker diarization pipelines"""

    @staticmethod
    def set_num_speakers(
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """Validate number of speakers

        Parameters
        ----------
        num_speakers : int, optional
            Number of speakers.
        min_speakers : int, optional
            Minimum number of speakers.
        max_speakers : int, optional
            Maximum number of speakers.

        Returns
        -------
        num_speakers : int or None
        min_speakers : int
        max_speakers : int or np.inf
        """

        # override {min|max}_num_speakers by num_speakers when available
        min_speakers = num_speakers or min_speakers or 1
        max_speakers = num_speakers or max_speakers or np.inf

        if min_speakers > max_speakers:
            raise ValueError(
                f"min_speakers must be smaller than (or equal to) max_speakers "
                f"(here: min_speakers={min_speakers:g} and max_speakers={max_speakers:g})."
            )
        if min_speakers == max_speakers:
            num_speakers = min_speakers

        return num_speakers, min_speakers, max_speakers

    # 必要ないのでコメントアウト
    # @staticmethod
    # def optimal_mapping(
    #     reference: Union[Mapping, Annotation],
    #     hypothesis: Annotation,
    #     return_mapping: bool = False,
    # ) -> Union[Annotation, Tuple[Annotation, Dict[Label, Label]]]:
    #     """Find the optimal bijective mapping between reference and hypothesis labels

    #     Parameters
    #     ----------
    #     reference : Annotation or Mapping
    #         Reference annotation. Can be an Annotation instance or
    #         a mapping with an "annotation" key.
    #     hypothesis : Annotation
    #         Hypothesized annotation.
    #     return_mapping : bool, optional
    #         Return the label mapping itself along with the mapped annotation. Defaults to False.

    #     Returns
    #     -------
    #     mapped : Annotation
    #         Hypothesis mapped to reference speakers.
    #     mapping : dict, optional
    #         Mapping between hypothesis (key) and reference (value) labels
    #         Only returned if `return_mapping` is True.
    #     """

    #     if isinstance(reference, Mapping):
    #         reference = reference["annotation"]
    #         annotated = reference["annotated"] if "annotated" in reference else None
    #     else:
    #         annotated = None

    #     mapping = DiarizationErrorRate().optimal_mapping(
    #         reference, hypothesis, uem=annotated
    #     )
    #     mapped_hypothesis = hypothesis.rename_labels(mapping=mapping)

    #     if return_mapping:
    #         return mapped_hypothesis, mapping

    #     else:
    #         return mapped_hypothesis
    

    # TODO: get rid of warm-up parameter (trimming should be applied before calling speaker_count)
    @staticmethod
    def speaker_count(
        binarized_segmentations: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        """Estimate frame-level number of instantaneous speakers

        Parameters
        ----------
        binarized_segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped binarized scores.
        warm_up : (float, float) tuple, optional
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.
        frames : SlidingWindow
            Frames resolution. Defaults to estimate it automatically based on
            `segmentations` shape and chunk size. Providing the exact frame
            resolution (when known) leads to better temporal precision.

        Returns
        -------
        count : SlidingWindowFeature
            (num_frames, 1)-shaped instantaneous speaker count
        """
        
        # print(binarized_segmentations.data.shape) #(21, 589, 3)
        # print(binarized_segmentations.sliding_window) #<pyannote_audio_utils.core.segment.SlidingWindow object at 0x3138911e0>
        # print(binarized_segmentations.sliding_window.duration) #10.0
        # print(binarized_segmentations.sliding_window.step) #1.0
        # print(binarized_segmentations.sliding_window.start) #0.0
        # print(binarized_segmentations.sliding_window.end) #inf
        # print(warm_up) #(0.0, 0.0)
        trimmed = Inference.trim(binarized_segmentations, warm_up=warm_up)
        # print(trimmed) #<pyannote_audio_utils.core.feature.SlidingWindowFeature object at 0x3164ccdf0>
        # print(trimmed.data.shape) #(21, 589, 3) OK
        # print(trimmed.sliding_window.duration) #10.0
        # print(trimmed.sliding_window.step) #1.0
        # print(trimmed.sliding_window.start) #0.0
        # print(trimmed.sliding_window.end) #inf
        
        # デバック用に追加
        # debug_sum_trimmed = np.sum(trimmed, axis=-1, keepdims=True)
        # print(debug_sum_trimmed.data.shape) #(21, 589, 1)
        # print(debug_sum_trimmed.sliding_window.duration) #10.0
        # print(debug_sum_trimmed.sliding_window.step) #1.0
        # print(debug_sum_trimmed.sliding_window.start) #0.0
        # print(debug_sum_trimmed.sliding_window.end) #inf
        # print(debug_sum_trimmed.data[0]) #最初0，途中から1 OK
        # print(debug_sum_trimmed.data[1]) #最初0，途中から1，途中から2 OK
        
        
        count = Inference.aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        # print(count.data.shape) #(1767, 1)
        # print(count.sliding_window.duration) #0.01697792869269949
        # print(count.sliding_window.step) #0.01697792869269949
        # print(count.sliding_window.start) #0.0
        # print(count.sliding_window.end) #inf
        
        
        
        
        count.data = np.rint(count.data).astype(np.uint8)
        
        # write_array_to_csv("count_data.csv", count.data) #OK
        

        return count

    @staticmethod
    def to_annotation(
        discrete_diarization: SlidingWindowFeature,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> Annotation:
        """

        Parameters
        ----------
        discrete_diarization : SlidingWindowFeature
            (num_frames, num_speakers)-shaped discrete diarization
        min_duration_on : float, optional
            Defaults to 0.
        min_duration_off : float, optional
            Defaults to 0.

        Returns
        -------
        continuous_diarization : Annotation
            Continuous diarization, with speaker labels as integers,
            corresponding to the speaker indices in the discrete diarization.
        """

        binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        # print(binarize) #<pyannote_audio_utils.audio.utils.signal.Binarize object at 0x31e9ddd20>
        # print(binarize.onset) #0.5
        # print(binarize.offset) #0.5
        # print(binarize.min_duration_on) #0.0
        # print(binarize.min_duration_off) #0.0
        # print(binarize.pad_onset) #0.0
        # print(binarize.pad_offset) #0.0
        
        
        
        # print(discrete_diarization) #<pyannote_audio_utils.core.feature.SlidingWindowFeature object at 0x32745fe20>
        # print(type(discrete_diarization.sliding_window[0])) #<class 'pyannote_audio_utils.core.segment.Segment'>
        # print(discrete_diarization.sliding_window[0]) #[ 00:00:00.000 -->  00:00:00.016]
        
        
        # print(binarize(discrete_diarization))
        # [ 00:00:06.714 -->  00:00:07.003] 2 2
        # [ 00:00:07.003 -->  00:00:07.173] 0 0
        # [ 00:00:07.580 -->  00:00:07.597] 0 0
        # [ 00:00:07.597 -->  00:00:08.276] 2 2
        # [ 00:00:08.276 -->  00:00:08.293] 0 0
        # [ 00:00:08.293 -->  00:00:08.310] 2 2
        # [ 00:00:08.310 -->  00:00:09.906] 0 0
        # [ 00:00:09.906 -->  00:00:10.959] 2 2
        # [ 00:00:10.466 -->  00:00:14.745] 0 0
        # [ 00:00:10.959 -->  00:00:10.976] 1 1
        # [ 00:00:14.303 -->  00:00:17.886] 1 1
        # [ 00:00:18.022 -->  00:00:21.502] 0 0
        # [ 00:00:18.157 -->  00:00:18.446] 1 1
        # [ 00:00:21.774 -->  00:00:28.531] 1 1
        # [ 00:00:27.886 -->  00:00:29.991] 0 0
        
        # print(binarize(discrete_diarization).rename_tracks(generator="string"))
        # [ 00:00:06.714 -->  00:00:07.003] A 2
        # [ 00:00:07.003 -->  00:00:07.173] B 0
        # [ 00:00:07.580 -->  00:00:07.597] C 0
        # [ 00:00:07.597 -->  00:00:08.276] D 2
        # [ 00:00:08.276 -->  00:00:08.293] E 0
        # [ 00:00:08.293 -->  00:00:08.310] F 2
        # [ 00:00:08.310 -->  00:00:09.906] G 0
        # [ 00:00:09.906 -->  00:00:10.959] H 2
        # [ 00:00:10.466 -->  00:00:14.745] I 0
        # [ 00:00:10.959 -->  00:00:10.976] J 1
        # [ 00:00:14.303 -->  00:00:17.886] K 1
        # [ 00:00:18.022 -->  00:00:21.502] L 0
        # [ 00:00:18.157 -->  00:00:18.446] M 1
        # [ 00:00:21.774 -->  00:00:28.531] N 1
        # [ 00:00:27.886 -->  00:00:29.991] O 0

        

        return binarize(discrete_diarization).rename_tracks(generator="string")

    @staticmethod
    def to_diarization(
        segmentations: SlidingWindowFeature,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build diarization out of preprocessed segmentation and precomputed speaker count

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_speakers)-shaped segmentations
        count : SlidingWindow_feature
            (num_frames, 1)-shaped speaker count

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """
        
        
        # write_array_to_csv("count.csv", count.data) #OK
         
         

        # TODO: investigate alternative aggregation
        activations = Inference.aggregate(
            segmentations,
            count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )
        # shape is (num_frames, num_speakers)
        
        # print(activations.data.shape) #(1767, 3)
        # print(activations.sliding_window.duration) #0.01697792869269949
        # print(activations.sliding_window.step) #0.01697792869269949
        # print(activations.sliding_window.start) #0.0
        # print(activations.sliding_window.end) #inf
        # write_array_to_csv("activations.csv", activations.data)
        
        
        

        _, num_speakers = activations.data.shape
        max_speakers_per_frame = np.max(count.data)
        if num_speakers < max_speakers_per_frame:
            activations.data = np.pad(
                activations.data, ((0, 0), (0, max_speakers_per_frame - num_speakers))
            )
        # print(num_speakers) #3
        # print(max_speakers_per_frame) #2
            

        # print(activations.extent) #[ 00:00:00.000 -->  00:00:30.000]
        # print(activations.extent.start) #0.0
        # print(activations.extent.end) #30.0
        # print(count.extent.start) #0.0
        # print(count.extent.end) #30.0
        extent = activations.extent & count.extent
        # print(extent.start) #0.0
        # print(extent.end) #30.0
        
        activations = activations.crop(extent, return_data=False)
        # print(activations.data.shape) #(1767, 3)
        # print(activations.sliding_window.duration) #0.01697792869269949
        # print(activations.sliding_window.step) #0.01697792869269949
        # print(activations.sliding_window.start) #0.0
        # print(activations.sliding_window.end) #inf
        # write_array_to_csv("activations.csv", activations.data)
        
        
        count = count.crop(extent, return_data=False)
        # print(count.data.shape) #(1767, 1)
        # print(count.sliding_window.duration) #0.01697792869269949
        # print(count.sliding_window.step) #0.01697792869269949
        # print(count.sliding_window.start) #0.0
        # print(count.sliding_window.end) #inf
        # write_array_to_csv("counts.csv", count.data)
        
        
        

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)
        # print(sorted_speakers.shape) #(1767, 3)
        # print(binary.shape) #(1767, 3)
        # write_array_to_csv("sorted_speakers.csv", sorted_speakers)
        # write_array_to_csv("binary.csv", binary)

        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0
                
        # write_array_to_csv("binary.csv", binary)

        return SlidingWindowFeature(binary, activations.sliding_window)

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1
