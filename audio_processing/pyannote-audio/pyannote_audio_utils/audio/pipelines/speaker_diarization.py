# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Speaker diarization pipelines"""

import functools
import itertools
import math
import textwrap
import warnings
import numpy as np

from typing import Callable, Optional, Text, Union, Mapping
from pathlib import Path

from pyannote_audio_utils.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote_audio_utils.pipeline.parameter import ParamDict, Uniform
from pyannote_audio_utils.audio import Audio, Inference, Pipeline
from pyannote_audio_utils.audio.core.io import AudioFile
from pyannote_audio_utils.audio.pipelines.clustering import Clustering
from pyannote_audio_utils.audio.pipelines.speaker_verification import ONNXWeSpeakerPretrainedSpeakerEmbedding
from pyannote_audio_utils.audio.pipelines.utils import SpeakerDiarizationMixin

AudioFile = Union[Text, Path, Mapping]
PipelineModel = Union[Text, Mapping]


# デバック
## ========================================================================
import csv

def write_array_to_csv(filename, data):
    """2次元配列をCSVファイルに保存する関数

    Args:
        data (list of list): 保存する2次元配列
        filename (str): 保存するファイル名
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
## ========================================================================


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
# class SpeakerDiarization(Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote_audio_utils/segmentation@2022.07".
        See pyannote_audio_utils.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        The segmentation model is applied on a window sliding over the whole audio file.
        `segmentation_step` controls the step of this window, provided as a ratio of its
        duration. Defaults to 0.1 (i.e. 90% overlap between two consecuive windows).
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote_audio_utils/embedding@2022.07".
        See pyannote_audio_utils.audio.pipelines.utils.get_model for supported format.
    embedding_exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        Defaults (False) to use the whole speech.
    clustering : str, optional
        Clustering algorithm. See pyannote_audio_utils.audio.pipelines.clustering.Clustering
        for available options. Defaults to "AgglomerativeClustering".
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 1.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 1.
    der_variant : dict, optional
        Optimize for a variant of diarization error rate.
        Defaults to {"collar": 0.0, "skip_overlap": False}. This is used in `get_metric`
        when instantiating the metric: GreedyDiarizationErrorRate(**der_variant).
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    # perform (unconstrained) diarization
    >>> diarization = pipeline("/path/to/audio.wav")

    # perform diarization, targetting exactly 4 speakers
    >>> diarization = pipeline("/path/to/audio.wav", num_speakers=4)

    # perform diarization, with at least 2 speakers and at most 10 speakers
    >>> diarization = pipeline("/path/to/audio.wav", min_speakers=2, max_speakers=10)

    # perform diarization and get one representative embedding per speaker
    >>> diarization, embeddings = pipeline("/path/to/audio.wav", return_embeddings=True)
    >>> for s, speaker in enumerate(diarization.labels()):
    ...     # embeddings[s] is the embedding of speaker `speaker`

    Hyper-parameters
    ----------------
    segmentation.threshold
    segmentation.min_duration_off
    clustering.???
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote_audio_utils/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        args = None,
        seg_path = None, 
        emb_path = None,
        der_variant: dict = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()
        
        # print(segmentation_step) #0.1
        # print(args)
        # Namespace(input=['./data/sample.wav'], video=None, savepath=None, benchmark=False, env_id=2, env_list=False, ftype='audio', debug=False, 
        # profile=False, benchmark_count=5, num=0, max=0, min=0, ig=None, o='output.png', og='output_ground.png', e=False, plt=False, embed=False, onnx=False)
        # print(args.num) #3
        
        model = segmentation
        self.segmentation_step = segmentation_step
        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap
        self.klustering = clustering
        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        segmentation_duration = 10.0
            
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
            args=args,
            seg_path=seg_path
        )
        
        # print(self._segmentation) #<pyannote_audio_utils.audio.core.inference.Inference object at 0x16bb50ac0>
        # print(self._segmentation.duration) #10.0
        # print(self._segmentation.step) #1.0
        
        
        self._frames: SlidingWindow = self._segmentation.example_output.frames
        # print(self._frames.duration) #0.01697792869269949
        # print(self._frames.step) #0.01697792869269949
        # print(self._frames.start) #0.0
        # print(self._frames.end) #inf
        
        self.segmentation = ParamDict(
            min_duration_off=Uniform(0.0, 1.0),
        )
        
        self._embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding(
            self.embedding, 
            args=args,
            emb_path=emb_path
        )
        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")

        metric = self._embedding.metric  
        # print(metric)s #cosine    
        Klustering = Clustering[clustering]

        self.clustering = Klustering.value(metric=metric)


    def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile
        hook : Optional[Callable]

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """
        
        if hook is not None:
            hook = functools.partial(hook, "segmentation", None)
            
        # print(file) #{'audio': './data/sample.wav', 'uri': 'sample'}
        segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook)
        
        # print(segmentations.data.shape) #(21, 589, 3)
        # print(segmentations.sliding_window.duration) #10.0
        # print(segmentations.sliding_window.step) #1.0
        # print(segmentations.sliding_window.start) #0.0
        # print(segmentations.sliding_window.end) #inf
        
        # print(segmentations)

        return segmentations

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
        hook: Optional[Callable] = None,
    ):
        """Extract embeddings for each (chunk, speaker) pair

        Parameters
        ----------
        file : AudioFile
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Binarized segmentation.
        exclude_overlap : bool, optional
            Exclude overlapping speech regions when extracting embeddings.
            In case non-overlapping speech is too short, use the whole speech.
        hook: Optional[Callable]
            Called during embeddings after every batch to report the progress

        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) array
        """

        # when optimizing the hyper-parameters of this pipeline with frozen
        # "segmentation.threshold", one can reuse the embeddings from the first trial,
        # bringing a massive speed up to the optimization process (and hence allowing to use
        # a larger search space).
   
        duration = binary_segmentations.sliding_window.duration
        # print(duration) #10.0
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape
        # print(binary_segmentations.data.shape) #(21, 589, 3)

        if exclude_overlap:
            
            # minimum number of samples needed to extract an embedding
            # (a lower number of samples would result in an error)
            min_num_samples = self._embedding.min_num_samples

            # corresponding minimum number of frames
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            # zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )
        
        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )
            
        # for chunk, masks in binary_segmentations:
            # print(chunk) #[ 00:00:00.000 -->  00:00:10.000], [ 00:00:01.000 -->  00:00:11.000], [ 00:00:02.000 -->  00:00:12.000]
            # print(masks) #speakerのラベルが入っている
            # print(masks.shape) #(589, 3), (589, 3), (589, 3) ...
        
        # print(binary_segmentations.data.shape) #(21, 589, 3)
        # print(clean_segmentations.data.shape) #(21, 589, 3)
        
        # write_array_to_csv("clean_segmentations.csv", clean_segmentations.data.reshape(-1, 3))
        
        def iter_waveform_and_mask():
            
            # デバック用変数
            all_masks = None  # 全てのmasksを格納するための変数を初期化
            all_clean_masks = None  # 全てのmasksを格納するための変数を初期化
            all_used_masks = None  # 全てのmasksを格納するための変数を初期化
            
            
            for (chunk, masks), (_, clean_masks) in zip(binary_segmentations, clean_segmentations): #このfor文は21回呼ばれる
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                # print(chunk) #[ 00:00:00.000 -->  00:00:10.000], [ 00:00:01.000 -->  00:00:11.000], [ 00:00:02.000 -->  00:00:12.000]
                # print(duration) #10.0, 10.0, 10.0, 10.0, 10.0
                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # print(waveform.shape) #(1, 160000),(1, 160000),(1, 160000),(1, 160000)... 21個続く
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
                
                # print(masks.shape) #(589, 3), (589, 3), (589, 3) ... 21個
                # デバック===============================================================
                if all_masks is None:
                    # all_masksが初期化されていなければ、最初のmasksで初期化
                    all_masks = masks.T  # 転置しておく (重要: 後で簡単に結合できるように)
                    all_clean_masks = clean_masks.T
                else:
                    # 以降はmasksを転置してall_masksに結合
                    all_masks = np.concatenate((all_masks, masks.T), axis=0)
                    all_clean_masks = np.concatenate((all_clean_masks, clean_masks.T), axis=0)
                # print(all_masks.shape) #(3, 589), (6, 589), (9, 589) ... (63, 589)
                # write_array_to_csv("masks.csv", all_masks) #OK
                # write_array_to_csv("clean_masks.csv", all_clean_masks) #OK
                # ======================================================================

                for mask, clean_mask in zip(masks.T, clean_masks.T): #このfor文は3回呼ばれる
                    # mask: (num_frames, ) np.ndarray
                    # print(mask.shape) #(589,), (589,), (589,), (589,), (589,)...
                    # print(clean_mask.shape) #(589,), (589,), (589,), (589,), (589,)...

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                        # print("clean_mask") こっちが2回呼ばれる
                    else:
                        used_mask = mask
                        # print("mask") こっちが1回呼ばれる
                        
                    # print(waveform.shape) #(1, 160000),(1, 160000),(1, 160000)...
                    # print(waveform[None].shape) #(1, 1, 160000), (1, 1, 160000), (1, 1, 160000), (1, 1, 160000),  ///

                    # デバック===============================================================
                    if all_used_masks is None:
                        # all_masksが初期化されていなければ、最初のmasksで初期化
                        all_used_masks = used_mask[None]  # 転置しておく (重要: 後で簡単に結合できるように)
                    else:
                        # 以降はmasksを転置してall_masksに結合
                        all_used_masks = np.concatenate((all_used_masks, used_mask[None]), axis=0)
                    # print(all_used_masks.shape) #(3, 589), (6, 589), (9, 589) ... (63, 589)
                    # write_array_to_csv("used_masks.csv", all_used_masks) #OK
                    # ======================================================================
                        
                    
                    yield waveform[None], used_mask[None]
                    # waveformにはチャンクで区切った音声データが入っている
                    # used_maskには[0,0,1]のように話しているspeakerにフラグが立ったsegmentationの結果が入っている
                    
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )
        
        # print(batches) #<itertools.zip_longest object at 0x30b0eeb60>
        

        batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)
        # バッチサイズは32
        # audioファイルをwindowで区切ったのは21
        # そこにnum_speakers = 3をかけて63になっている
        # print(batch_count) #2

        embedding_batches = []

        if hook is not None:
            hook("embeddings", None, total=batch_count, completed=0)

        # print(batches) #<itertools.zip_longest object at 0x31309ed90>
        # print(self.embedding_batch_size) #32
        
        # デバック
        # print("デバック")
        # debug_waveform = np.arange(1, 160001).reshape(1, 1, 160000)
        # debug_masks = np.ones((1, 589))
        # debug_embedding_batch: np.ndarray = self._embedding(debug_waveform, debug_masks)
        # print(debug_embedding_batch.shape) #(1, 256)
        # debug_embeddings = debug_embedding_batch.reshape([num_chunks, -1 , debug_embedding_batch.shape[-1]])
        # print(debug_embeddings.shape)
        
        
        for i, batch in enumerate(batches, 1):
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))
            # print(len(waveforms)) #32 ,31
            # print(waveforms[0].shape) #(1, 1, 160000)
            # print(waveforms[1].shape) #(1, 1, 160000)
            # print(waveforms[2].shape) #(1, 1, 160000)
            # print(waveforms[0]) #[[ 0. 0. 0. ... -0.021698 -0.02038574 -0.01696777]]], [[[-1.19323730e-02 -6.95800781e-03 -2.59399414e-03 ... -3.66210938e-04 -9.5527344e-05 4.27246094e-04]]]
            # print(waveforms[1]) #[[ 0. 0. 0. ... -0.021698 -0.02038574 -0.01696777]]], [[[-1.19323730e-02 -6.95800781e-03 -2.59399414e-03 ... -3.66210938e-04 -9.5527344e-05 4.27246094e-04]]]
            # print(waveforms[2]) #[[ 0. 0. 0. ... -0.021698 -0.02038574 -0.01696777]]], [[[-1.19323730e-02 -6.95800781e-03 -2.59399414e-03 ... -3.66210938e-04 -9.5527344e-05 4.27246094e-04]]]
            # この3つは全て同じ配列
            # print(waveforms[3]) #次の3つも同じ
            # print(waveforms[4])
            # print(waveforms[5])
            
            # print(len(masks)) #32, 31
            # print(masks[0].shape) #(1, 589)
            # print(masks[1].shape) #(1, 589)
            # print(masks[2].shape) #(1, 589)
            # print(masks[0]) #全部0 #maskにはspeaker1からspeaker3までの話しているかどうかのフラグが入っていそう
            # print(masks[1]) #後半だけ1
            # print(masks[2]) #全部0
            
            waveform_batch = np.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = np.vstack(masks)
            # (batch_size, num_frames) torch.Tensor
            
            # print(waveform_batch.shape) #(32, 1, 160000), (31, 1, 160000)
            # print(mask_batch.shape) #(32, 589), (31, 589)
            
            # ここで推論
            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            # (batch_size, dimension) np.ndarray
            # print(embedding_batch.shape) #(32, 256), (31, 256)

            embedding_batches.append(embedding_batch)

            if hook is not None:
                hook("embeddings", embedding_batch, total=batch_count, completed=i)
        
        # print(len(embedding_batches)) #2
        
        embedding_batches = np.vstack(embedding_batches)
        # print(embedding_batches.shape) #(63, 256)
        # write_array_to_csv("embedding_batches.csv", embedding_batches) #C++と違う
        
        embeddings = embedding_batches.reshape([num_chunks, -1 , embedding_batches.shape[-1]])
        # print(embeddings.shape) #(21, 3, 256)

        return embeddings

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        
        
        # print(segmentations.data.shape) #(21, 589, 3)
        # print(segmentations.sliding_window.duration) #10.0
        # print(segmentations.sliding_window.step) #1.0
        # print(segmentations.sliding_window.start) #0.0
        # print(segmentations.sliding_window.end) #inf
        
        # print(hard_clusters.shape) #(21, 3)
        # write_array_to_csv("hard_clusters.csv", hard_clusters) #
        
        # print(count.data.shape) #(1767, 1)
        # print(count.sliding_window.duration) #0.01697792869269949
        # print(count.sliding_window.step) #0.01697792869269949
        # print(count.sliding_window.start) #0.0
        # print(count.sliding_window.end) #inf
        
        """Build final discrete diarization out of clustered segmentation

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        hard_clusters : (num_chunks, num_speakers) array
            Output of clustering step.
        count : (total_num_frames, 1) SlidingWindowFeature
            Instantaneous number of active speakers.

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )
        
        

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):
            # print(c)
            # print(cluster)
            # print(segmentation)
            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )
                
                
        # print(clustered_segmentations.shape) #(21, 589, 3)
        # write_array_to_csv("clustered_segmentations[0].csv", clustered_segmentations[0])
        # write_array_to_csv("clustered_segmentations[1].csv", clustered_segmentations[1])
        # write_array_to_csv("clustered_segmentations[2].csv", clustered_segmentations[2])
        # write_array_to_csv("clustered_segmentations[20].csv", clustered_segmentations[20]) #C++と違う
        
        

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )
        
        # debug_result = self.to_diarization(clustered_segmentations, count)
        # print(debug_result) #<pyannote_audio_utils.core.feature.SlidingWindowFeature object at 0x3168dd2a0>
        # print(debug_result.data) #floatの配列
        # print(debug_result.data.shape) #(1767, 3)
        # print(debug_result.sliding_window.duration) #0.01697792869269949
        # print(debug_result.sliding_window.step) #0.01697792869269949
        # print(debug_result.sliding_window.start) #0.0
        # print(debug_result.sliding_window.end) #inf

        return self.to_diarization(clustered_segmentations, count)

    def apply(
        self,
        file: AudioFile,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        return_embeddings: bool = False,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        return_embeddings : bool, optional
            Return representative speaker embeddings.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        embeddings : np.array, optional
            Representative speaker embeddings such that `embeddings[i]` is the
            speaker embedding for i-th speaker in diarization.labels().
            Only returned when `return_embeddings` is True.
        """
        
        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)
        
        # print(num_speakers) #None
        # print(min_speakers) #None
        # print(max_speakers) #None
        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        # print(num_speakers) #None
        # print(min_speakers) #1
        # print(max_speakers) #inf
        
        
        segmentations = self.get_segmentations(file, hook=hook)
        # print(segmentations.data.shape) #(21, 589, 3) #(2, 589, 3)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)

        # binarize segmentation
        
        binarized_segmentations = segmentations
        # write_array_to_csv("multilabel_encoding_segmentation.csv", binarized_segmentations.data.reshape(-1, 3))
        # write_array_to_csv("reshape_segmentation_20.csv", binarized_segmentations[20])
        
        # print(binarized_segmentations.data.shape) #(21, 589, 3)
        # print(self._frames) #<pyannote_audio_utils.core.segment.SlidingWindow object at 0x16b9dd1e0>
        # print(self._frames.duration) #0.01697792869269949
        # print(self._frames.step) #0.01697792869269949
        # print(self._frames.start) #0.0
        # print(self._frames.end) #inf
        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            frames=self._frames,
            warm_up=(0.0, 0.0),
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int
        # print(count.data.shape) #(1767, 1)
        save_data(count.data, "python_data/speaker_count.json")

        # exit early when no speaker is ever active
        if np.nanmax(count.data) == 0.0:
            diarization = Annotation(uri=file["uri"])
            if return_embeddings:
                return diarization, np.zeros((0, self._embedding.dimension))

            return diarization

        
        # print(binarized_segmentations.data.shape) #(21, 589, 3)
        save_data(binarized_segmentations.data, "python_data/binarized_segmentations.json")

        embeddings = self.get_embeddings(
            file,
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
            hook=hook,
        )
        
        # print(embeddings.shape) #(21, 3, 256)
        save_data(embeddings, "python_data/embeddings.json")
        
        hook("embeddings", embeddings)
        #   shape: (num_chunks, local_num_speakers, dimension)
        
        # print(num_speakers) #None
        # print(min_speakers) #1
        # print(max_speakers) #inf
        
        # -num 3 を指定した場合
        # print(num_speakers) #3
        # print(min_speakers) #3
        # print(max_speakers) #3

        hard_clusters, _, centroids = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,  # <== for oracle clustering
            frames=self._frames,  # <== for oracle clustering
        )
        # hard_clusters: (num_chunks, num_speakers)
        # centroids: (num_speakers, dimension)
        save_data(hard_clusters, "python_data/hard_clusters.json")
        save_data(_, "python_data/soft_clusters.json")
        save_data(centroids, "python_data/centroids.json")

        # number of detected clusters is the number of different speakers
        num_different_speakers = np.max(hard_clusters) + 1
        # print(num_different_speakers) #3 #1 #4
        
        
        # print(min_speakers) #4
        # print(max_speakers) #4
        # detected number of speakers can still be out of bounds
        # (specifically, lower than `min_speakers`), since there could be too few embeddings
        # to make enough clusters with a given minimum cluster size.
        if num_different_speakers < min_speakers or num_different_speakers > max_speakers:
            warnings.warn(textwrap.dedent(
                f"""
                The detected number of speakers ({num_different_speakers}) is outside
                the given bounds [{min_speakers}, {max_speakers}]. This can happen if the
                given audio file is too short to contain {min_speakers} or more speakers.
                Try to lower the desired minimal number of speakers.
                """
            ))

        # print(count) #<pyannote_audio_utils.core.feature.SlidingWindowFeature object at 0x16e7d9090>
        # print(count.data.shape) #(1767, 1)
        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, max_speakers).astype(np.int8)
        # print(count.data.shape) #(1767, 1) #OK
        # write_array_to_csv("count_data.csv", count.data) #OK

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        # print(binarized_segmentations.data.shape) #(21, 589, 3)
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)
        # print(inactive_speakers.shape) #(21, 3) OK
        # print(inactive_speakers) #true,faleの列 OK
        # write_array_to_csv("inactive_speakers.csv", inactive_speakers) #OK
        
        

        hard_clusters[inactive_speakers] = -2
        # print(hard_clusters.shape) #(21, 3) OK
        # write_array_to_csv("hard_clusters.csv", hard_clusters) #OK
        
        save_data(segmentations.data, "python_data/segmentations.json")
        # print(segmentations.sliding_window.duration) #10.0
        # print(segmentations.sliding_window.step) #1.0
        # print(segmentations.sliding_window.start) #0.0
        # print(segmentations.sliding_window.end) #inf
        save_data(hard_clusters, "python_data/hard_clusters.json")
        save_data(count.data, "python_data/count.json")
        # print(count.sliding_window.duration) #0.01697792869269949
        # print(count.sliding_window.step) #0.01697792869269949
        # print(count.sliding_window.start) #0.0
        # print(count.sliding_window.end) #inf
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)
        # print(discrete_diarization) #<pyannote_audio_utils.core.feature.SlidingWindowFeature object at 0x3281dd2a0>
        # print(discrete_diarization.data.shape) #(1767, 3)
        # print(discrete_diarization.sliding_window.duration) #0.01697792869269949
        # print(discrete_diarization.sliding_window.step) #0.01697792869269949
        # print(discrete_diarization.sliding_window.start) #0.0
        # print(discrete_diarization.sliding_window.end) #inf
        # write_array_to_csv("discrete_diarization.csv", discrete_diarization.data) #/2列目と3列目が入れ替わってるけど良さそう
        # print(discrete_diarization.sliding_window[0]) # [ 00:00:00.000 -->  00:00:00.016]
        # print(discrete_diarization.sliding_window[1]) # [ 00:00:00.016 -->  00:00:00.033]
        # print(discrete_diarization.sliding_window[2]) #[ 00:00:00.033 -->  00:00:00.050]
        # print(len(discrete_diarization.sliding_window)) #ValueError: infinite sliding window.
        save_data(discrete_diarization.data, "python_data/discrete_diarization.json")
        
        
        # print(self.segmentation.min_duration_off) #0.0

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]
        # print(diarization)
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
        
        

        # at this point, `diarization` speaker labels are integers
        # from 0 to `num_speakers - 1`, aligned with `centroids` rows.
        
        if "annotation" in file and file["annotation"]:
            # when reference is available, use it to map hypothesized speakers
            # to reference speakers (this makes later error analysis easier
            # but does not modify the actual output of the diarization pipeline)
            _, mapping = self.optimal_mapping(
                file["annotation"], diarization, return_mapping=True
            )

            # in case there are more speakers in the hypothesis than in
            # the reference, those extra speakers are missing from `mapping`.
            # we add them back here
            mapping = {key: mapping.get(key, key) for key in diarization.labels()}

        else:
            # print("here") #here
            # when reference is not available, rename hypothesized speakers
            # to human-readable SPEAKER_00, SPEAKER_01, ...
            mapping = {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }
            
            # print(diarization.labels()) #[0, 1, 2]
            # print(self.classes()) #<generator object SpeakerDiarizationMixin.classes at 0x30856c5f0>

        diarization = diarization.rename_labels(mapping=mapping)
        
        # print(diarization)
        # [ 00:00:06.714 -->  00:00:07.003] A SPEAKER_02
        # [ 00:00:07.003 -->  00:00:07.173] B SPEAKER_00
        # [ 00:00:07.580 -->  00:00:07.597] C SPEAKER_00
        # [ 00:00:07.597 -->  00:00:08.276] D SPEAKER_02
        # [ 00:00:08.276 -->  00:00:08.293] E SPEAKER_00
        # [ 00:00:08.293 -->  00:00:08.310] F SPEAKER_02
        # [ 00:00:08.310 -->  00:00:09.906] G SPEAKER_00
        # [ 00:00:09.906 -->  00:00:10.959] H SPEAKER_02
        # [ 00:00:10.466 -->  00:00:14.745] I SPEAKER_00
        # [ 00:00:10.959 -->  00:00:10.976] J SPEAKER_01
        # [ 00:00:14.303 -->  00:00:17.886] K SPEAKER_01
        # [ 00:00:18.022 -->  00:00:21.502] L SPEAKER_00
        # [ 00:00:18.157 -->  00:00:18.446] M SPEAKER_01
        # [ 00:00:21.774 -->  00:00:28.531] N SPEAKER_01
        # [ 00:00:27.886 -->  00:00:29.991] O SPEAKER_00
        
        
        
        # at this point, `diarization` speaker labels are strings (or mix of
        # strings and integers when reference is available and some hypothesis
        # speakers are not present in the reference)
        if not return_embeddings:
            return diarization

        # this can happen when we use OracleClustering
        if centroids is None:
            return diarization, None


        # print(diarization.labels()) #['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
        # The number of centroids may be smaller than the number of speakers
        # in the annotation. This can happen if the number of active speakers
        # obtained from `speaker_count` for some frames is larger than the number
        # of clusters obtained from `clustering`. In this case, we append zero embeddings
        # for extra speakers
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0)))

        # re-order centroids so that they match
        # the order given by diarization.labels()
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]

        return diarization, centroids





# デバック用に追加
# JSON形式でデータを保存
import json
import os
def save_data(array, filename):
    
    # 保存先のディレクトリを取得
    directory = os.path.dirname(filename)
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)


    # 元の形状情報を取得
    original_shape = array.shape

    # 配列をフラット化
    flattened_array = array.flatten().tolist()  # リストに変換

    # 保存するデータ構造を作成
    data = {
        "shape": original_shape,  # 元の形状情報
        "data": flattened_array   # フラット化されたデータ
    }

    # JSON形式で保存
    with open(filename, 'w') as json_file:
        # json.dump(data, json_file, indent=4)  # インデントを付けて読みやすく保存
        json.dump(data, json_file, separators=(',', ':'))  # 改行せずに保存
        # json.dump(data, json_file, separators=(',', ':'), indent=4)