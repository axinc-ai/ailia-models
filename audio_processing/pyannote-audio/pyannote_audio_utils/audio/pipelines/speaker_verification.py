# MIT License
#
# Copyright (c) 2021 CNRS
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


from functools import cached_property
from typing import Optional, Text, Union, Mapping

import numpy as np
import ailia

from pyannote_audio_utils.audio.pipelines.utils.kaldifeat import compute_fbank_feats
from pyannote_audio_utils.audio.core.inference import BaseInference

import csv

PipelineModel = Union[Text, Mapping]

# ---------- デバック ----------
def save_array_to_csv(filename, array):
    # 配列が2次元でない場合、2次元に変換する
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)
        
# ---------- デバック ----------



class ONNXWeSpeakerPretrainedSpeakerEmbedding(BaseInference):
    """Pretrained WeSpeaker speaker embedding

    Parameters
    ----------
    embedding : str
        Path to WeSpeaker pretrained speaker embedding
    device : torch.device, optional
        Device

    Usage
    -----
    >>> get_embedding = ONNXWeSpeakerPretrainedSpeakerEmbedding("hbredin/wespeaker-voxceleb-resnet34-LM")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert binary_masks.ndim == 1
    >>> assert binary_masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=binary_masks)
    """

    def __init__(
        self,
        embedding: Text = "hbredin/wespeaker-voxceleb-resnet34-LM",
        # device: Optional[torch.device] = None,
        args = None,
        emb_path = None
    ):
        # if not ONNX_IS_AVAILABLE:
            # raise ImportError(
                # f"'onnxruntime' must be installed to use '{embedding}' embeddings."
            # )

        super().__init__()

        # if not Path(embedding).exists():
            # try:
                # embedding = hf_hub_download(
                    # repo_id=embedding,
                    # filename="speaker-embedding.onnx",
                # )
            # except RepositoryNotFoundError:
                # raise ValueError(
                    # f"Could not find '{embedding}' on huggingface.co nor on local disk."
                # )

        self.embedding = embedding
        
        if args.onnx:
            import onnxruntime as ort
            #print("use onnx runtime")
            providers = ["CPUExecutionProvider", ("CUDAExecutionProvider",{"cudnn_conv_algo_search": "DEFAULT"})]

            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            self.session_ = ort.InferenceSession(
                embedding, sess_options=sess_options, providers=providers
            )
        else:
            #print("use ailia")

            self.session_ = ailia.Net(emb_path, weight=embedding, env_id=args.env_id)
        
        self.args = args
 
    @cached_property
    def sample_rate(self) -> int:
        return 16000

    @cached_property
    def dimension(self) -> int:
        # dummy_waveforms = np.random.rand(1, 1, 16000)
        dummy_waveforms = np.arange(1, 16001).reshape(1, 1, 16000)
        # print("dummy_waveforms")
        # print(dummy_waveforms)
        # print(dummy_waveforms.shape)
        features = self.compute_fbank(dummy_waveforms)
        # print("dummy_waveforms features.shape")
        # print(features.shape) #(1, 998, 80)
        # print(features)

        if self.args.onnx:
            embeddings = self.session_.run(output_names=["embs"], input_feed={"feats": features}
            )[0]
        else:
            embeddings = self.session_.predict([features])[0]
            
        # print(embeddings.shape) #(1, 256)

        _, dimension = embeddings.shape
        return dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        # print(self.sample_rate) #16000
        lower, upper = 2, round(0.5 * self.sample_rate)
        middle = (lower + upper) // 2
        while lower + 1 < upper:
            try:
                # print("middle waveform.shape")
                # print(np.random.randn(1, 1, middle).shape)
                features = self.compute_fbank(np.random.randn(1, 1, middle))
                # print("middle features.shape")
                # print(features.shape) #(1, 23, 80), (1, 11, 80), (1, 4, 80), (1, 7, 80), (1, 9, 80), (1, 8, 80)
                                    # (1, 9, 80), (1, 8, 80), (1, 8, 80), (1, 9, 80), (1, 8, 80), (1, 8, 80), (1, 8, 80)

            except AssertionError:
                # print("here") #呼ばれない
                lower = middle
                middle = (lower + upper) // 2
                continue

            if self.args.onnx:
                embeddings = self.session_.run(output_names=["embs"], input_feed={"feats": features})[0]
            else:
                embeddings = self.session_.predict([features])[0]
                
            # print(embeddings.shape) #(1, 256), (1, 256) ... (1, 256) 13回

            if np.any(np.isnan(embeddings)):
                # print("here") #8回呼ばれる
                lower = middle
            else:
                # print("here") #5回呼ばれる
                upper = middle
            middle = (lower + upper) // 2
            # print(middle) #2001, 1001, 1501, 1751, 1626, 1688, 1657, 1672, 1680, 1676, 1678, 1679, 1679

        # print(upper) #1680
        return upper

    @cached_property
    def min_num_frames(self) -> int:
        # print("min_num_frames waveform.shape")
        # print(np.random.randn(1, 1, self.min_num_samples).shape)
        # features = self.compute_fbank(np.random.randn(1, 1, self.min_num_samples))
        # print("min_num_frames features.shape")
        # print(features.shape)
        return self.compute_fbank(np.random.randn(1, 1, self.min_num_samples)).shape[1]
    

    def compute_fbank(
        self,
        waveforms: np.ndarray,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
    ) -> np.ndarray:
        """Extract fbank features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins)

        Source: https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py#L30C1-L50
        """
        
        # 左ビットシフト演算 32768倍にしている
        waveforms = waveforms * (1 << 15)
        
        # print(waveforms.shape) #(1, 1, 16000)など
        # for waveform in waveforms:
        #     print(waveform.shape) #dummyのときは(1, 16000)が一つ，それ以外の時は(1, 160000)が連続する
        
        # print(num_mel_bins) #80
        # print(frame_length) #25
        # print(frame_shift) #10
        # print(dither) #0.0
        # print(self.sample_rate) #16000

        ### ここで少しずれる ###
        features_numpy = np.stack([compute_fbank_feats(
            waveform=waveform[0],
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=self.sample_rate,
            window_type="hamming",
            use_energy=False,
        )for waveform in waveforms])
        # )for waveform in waveforms if print(f"Waveform shape: {waveform[0].shape}") or True])
        # ) if not print(f"Shape of features: {compute_fbank_feats(waveform=waveform[0], num_mel_bins=num_mel_bins, frame_length=frame_length, frame_shift=frame_shift, dither=dither, sample_frequency=self.sample_rate, window_type='hamming', use_energy=False).shape}") else None for waveform in waveforms])
        ### ここで少しずれる ###
        
        # if(waveforms[0][0][0] == 0.0):
            # print(waveforms[0]) #[[   0.    0.    0. ... -711. -668. -556.]]
            # print(waveforms.shape) #(32, 1, 160000)
            # print(features_numpy.shape) #(32, 998, 80)
            # save_array_to_csv("features0.csv", features_numpy[0]) #C++と違う #features0とfeatures1とfeatures2は同じ
            # save_array_to_csv("features1.csv", features_numpy[1]) #C++と違う #features0とfeatures1とfeatures2は同じ
            # save_array_to_csv("features2.csv", features_numpy[2]) #C++と違う #features0とfeatures1とfeatures2は同じ
            # save_array_to_csv("features3.csv", features_numpy[3]) #C++と違う #features3とfeatures4とfeatures5は同じ
            # save_array_to_csv("features4.csv", features_numpy[4]) #C++と違う #features3とfeatures4とfeatures5は同じ
            # save_array_to_csv("features5.csv", features_numpy[5]) #C++と違う #features3とfeatures4とfeatures5は同じ
            # save_array_to_csv()
        # elif(waveforms[0][0][0] == -391.0):
        #     # print(waveforms[0]) #[[-391. -228.  -85. ...  -12.   -3.   14.]]
        #     # print(waveforms.shape) #(31, 1, 160000)
        #     # print(features_numpy.shape) #(31, 998, 80)
            
        # print(waveforms[0][0][0:10]) 
        #[  0.   0.   0.  -3.  -8. -13. -14. -16. -17. -21.]と出てくるタイミングが，元のaudiofileに対応していそう
        # 上の時，features_numpyのshapeが(32, 998, 80)
        # または，[-391. -228.  -85.   73.  271.  489.  671.  793.  882.  960.]
        # 上の時，features_numpyのshapeが(31, 998, 80)
        # print("features_numpy.shape {}".format(features_numpy.shape))
        
        features = features_numpy.astype(np.float32)
        
        # if(waveforms[0][0][0] == 32768):
        #     save_array_to_csv("features.csv", features[0]) #OK
            # print(np.mean(features, axis=1, keepdims=True).shape) #(1, 1, 80) #OK
            # print(np.mean(features, axis=1, keepdims=True))
            # print(features - np.mean(features, axis=1, keepdims=True))
            # save_array_to_csv("features.csv", (features - np.mean(features, axis=1, keepdims=True))[0]) #OK
 
        # print(features - np.mean(features, axis=1, keepdims=True))
        
        # if(waveforms[0][0][0] == 0.0):
        #     print(features_numpy.shape) #(32, 998, 80)
        #     save_array_to_csv("features0.csv", (features - np.mean(features, axis=1, keepdims=True))[0]) #C++と違う #features0とfeatures1とfeatures2は同じ
        #     save_array_to_csv("features1.csv", (features - np.mean(features, axis=1, keepdims=True))[1]) #C++と違う #features0とfeatures1とfeatures2は同じ
        #     save_array_to_csv("features2.csv", (features - np.mean(features, axis=1, keepdims=True))[2]) #C++と違う #features0とfeatures1とfeatures2は同じ
        
        return features - np.mean(features, axis=1, keepdims=True)

    def __call__(
        self, waveforms: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1
        
        # print(waveforms.shape) #(32, 1, 160000), (31, 1, 160000)
        
        features = self.compute_fbank(waveforms)
        _, num_frames, _ = features.shape
        # print(features.shape) #(32, 998, 80), (31, 998, 80)
        # print(features[0][0][0]) #-3.948223, 3.0625858
        # if(features[0][0][0] > -3.948224 and features[0][0][0] < -3.948222):
        #     save_array_to_csv("features0.csv", features[0]) # OK
        #     save_array_to_csv("features1.csv", features[1]) # OK
        #     save_array_to_csv("features2.csv", features[2]) # OK
            
        # print(masks)#0と1が並んでいる
        # print(features.shape) #(32, 998, 80), (31, 998, 80)
        # print(masks.shape) #(32, 589), (31, 589)
        # if(features[0][0][0] > -3.948224 and features[0][0][0] < -3.948222):
        #     save_array_to_csv("masks.csv", masks) #OK
        # else:
        #     save_array_to_csv("masks2.csv", masks) #OK
            
        
        batch_size_masks, _ = masks.shape
        assert batch_size == batch_size_masks
        
        def interpolate_numpy(input_array, size):
            output_array = np.zeros((input_array.shape[0],size))
            
            for i in range(output_array.shape[0]):
                for j in range(output_array.shape[1]):
                    ii = int(np.floor(i * input_array.shape[0] / output_array.shape[0]))
                    jj = int(np.floor(j * input_array.shape[1] / output_array.shape[1]))
                    output_array[i, j] = input_array[ii, jj]
            return output_array

        imasks = interpolate_numpy(masks,size=num_frames)
        # if(features[0][0][0] > -3.948224 and features[0][0][0] < -3.948222):
        #     save_array_to_csv("imasks.csv", imasks)
        # else:
        #     save_array_to_csv("imasks2.csv", imasks)
        imasks = imasks > 0.5
        # print(imasks.shape) #(32, 998), (31, 998) 
        # print(imasks[0])
        
        embeddings = np.NAN * np.zeros((batch_size, self.dimension))
        # print(embeddings) #全部nanで形だけある配列
        # print(embeddings.shape) #(32, 256), (31, 256)「--num 2」でも「--num 4」でも変わらず
        
        # print(self.min_num_frames) #9
        for f, (feature, imask) in enumerate(zip(features, imasks)):
            # print("feature.shape {}".format(feature.shape)) (998, 80)
            # print("imask.shape {}".format(imask.shape)) (998,)
            masked_feature = feature[imask]
            # print(masked_feature.shape) #(0, 80),(330, 80),(0, 80),(0, 80),(205, 80), (154, 80), (0, 80), (294, 80), (142, 80)... OK
            # (0, 80), (330, 80), (0, 80), (0, 80), (205, 80), (154, 80)
            # (0, 80), (294, 80), (142, 80), (0, 80), (417, 80), (119, 80)
            # (0, 80), (455, 80), (179, 80), (0, 80), (494, 80), (208, 80)
            # (0, 80), (305, 80), (493, 80), (0, 80), (350, 80), (507, 80)
            # (0, 80), (351, 80), (522, 80), (0, 80), (351, 80), (515, 80)
            # (0, 80), (362, 80), (526, 80), (0, 80), (315, 80), (599, 80)
            # (0, 80), (338, 80), (556, 80), (0, 80), (440, 80), (443, 80)
            # (0, 80), (533, 80), (362, 80), (0, 80), (611, 80), (349, 80)
            # (0, 80), (613, 80), (309, 80), (0, 80), (613, 80), (279, 80)
            # (0, 80), (311, 80), (621, 80), (0, 80), (609, 80), (300, 80)
            # (0, 80), (606, 80), (295, 80)
            
            # if(features[0][0][0] > -3.948224 and features[0][0][0] < -3.948222):
            #     if(f == 5):
                    # save_array_to_csv("masked_feature1.csv", masked_feature) #OK
                    # save_array_to_csv("masked_feature4.csv", masked_feature) #OK
                    # save_array_to_csv("masked_feature5.csv", masked_feature) #OK
            if masked_feature.shape[0] < self.min_num_frames:
                continue
            
            # print("masked_feature.shape {}".format(masked_feature.shape)) #(330, 80), (205, 80), (154, 80), (294, 80), (142, 80) OK
            
            
            if self.args.onnx:
                embeddings[f] = self.session_.run(output_names=["embs"],input_feed={"feats": masked_feature[None]},)[0][0]
            else:
                # print("masked_feature[None].shape {}".format(masked_feature[None].shape)) #(1, 330, 80),(1, 205, 80),(1, 154, 80),(1, 294, 80)
                embeddings[f] = self.session_.predict([masked_feature[None]])[0][0]
                
        # print(embeddings.shape) #(32, 256), (31, 256)
        
        # if(features[0][0][0] > -3.948224 and features[0][0][0] < -3.948222):
        #     save_array_to_csv("embeddings.csv", embeddings) #C++と違う
        # else:
        #     save_array_to_csv("embeddings2.csv", embeddings) #C++と違う
            

        return embeddings

