import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402

import torch
from torchvision import transforms

import onnxruntime

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='CRNN Audio Classification.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)

args = parser.parse_args()


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "crnn.onnx"
MODEL_PATH = "crnn.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crnn/"

# ======================
# Preprocess
# ======================

import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm
from torchaudio.transforms import TimeStretch, AmplitudeToDB 
from torch.distributions import Uniform

def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length

class MelspectrogramStretch(MelSpectrogram):

    def __init__(self, hop_length=None, 
                       sample_rate=44100, 
                       num_mels=128, 
                       fft_length=2048, 
                       norm='whiten', 
                       stretch_param=[0.4, 0.4]):

        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate, 
                                                    n_fft=fft_length, 
                                                    hop_length=hop_length, 
                                                    n_mels=num_mels)

        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length, pad=self.pad, 
                                       power=None, normalized=False)

        # Augmentation
        self.prob = stretch_param[0]
        #self.random_stretch = RandomTimeStretch(stretch_param[1], 
        #                                        self.hop_length, 
        #                                        self.n_fft//2+1, 
        #                                        fixed_rate=None)
        
        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.)
        self.norm = SpecNormalization(norm)

    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft//2)
            lengths = lengths.long()
        
        #if torch.rand(1)[0] <= self.prob and self.training:
        #    # Stretch spectrogram in time using Phase Vocoder
        #    x, rate = self.random_stretch(x)
        #    # Modify the rate accordingly
        #    lengths = (lengths.float()/rate).long()+1
        
        x = self.complex_norm(x)
        x = self.mel_scale(x)

        # Normalize melspectrogram
        x = self.norm(x)

        if lengths is not None:
            return x, lengths        
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):

        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x
        
    
    def z_transform(self, x):
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 
        return x

    def forward(self, x):
        return self._norm(x)

class AudioTransforms(object):

    def __init__(self):
        
        self.transfs = {
            'val': transforms.Compose([
                ProcessChannels("avg"),
                ToTensorAudio()
            ])#,
            #'train': transforms.Compose([
            #    ProcessChannels(args['channels']),
            #    AdditiveNoise(*args['noise']),
            #    RandomCropLength(*args['crop']),
            #    ToTensorAudio()
            #])
        }["val"]
        
    def apply(self, data, target):
        audio, sr = data
        # audio -> (time, channel)
        return self.transfs(audio), sr, target
        
    def __repr__(self):
        return self.transfs.__repr__()


class ProcessChannels(object):

    def __init__(self, mode):
        self.mode = mode

    def _modify_channels(self, audio, mode):
        if mode == 'mono':
            new_audio = audio if audio.ndim == 1 else audio[:,:1]
        elif mode == 'stereo':
            new_audio = np.stack([audio]*2).T if audio.ndim == 1 else audio
        elif mode == 'avg':
            new_audio= audio.mean(axis=1) if audio.ndim > 1 else audio
            new_audio = new_audio[:,None] 
        else:
            new_audio = audio
        return new_audio

    def __call__(self, tensor):
        return self._modify_channels(tensor, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(mode={})'.format(self.mode)


class ToTensorAudio(object):

    def __call__(self, tensor):
        return torch.from_numpy(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'

import soundfile as sf

def load_audio(path):
    return sf.read(path)

def preprocess(batch):
    spec = MelspectrogramStretch(hop_length=None, 
                            num_mels=128, 
                            fft_length=2048, 
                            norm='whiten', 
                            stretch_param=[0.4, 0.4])
                            
    # x-> (batch, time, channel)
    x, lengths, _ = batch # unpacking seqs, lengths and srs
    # x-> (batch, channel, time)
    xt = x.float().transpose(1,2)
    # xt -> (batch, channel, freq, time)
    print(xt.shape)
    xt, lengths = spec(xt, lengths)                
    print(xt.shape)

    #for key in self.net:
    #    print(key)
    
    return xt, lengths


def postprocess(out_raw):
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    out = torch.exp(out_raw)
    max_ind = out.argmax().item()      
    print(max_ind)  
    print(out.shape)
    print(len(classes))
    return classes[max_ind], out[:,max_ind].item()

# ======================
# Main function
# ======================

def crnn(path):
    # normal inference
    data = load_audio(path)
    tsf = AudioTransforms()
    sig_t, sr, _ = tsf.apply(data, None)

    length = torch.tensor(sig_t.size(0))
    sr = torch.tensor(sr)
    data = [d.unsqueeze(0).to("cpu") for d in [sig_t, length, sr]]
    
    #label, conf = self.model.predict( data )

    xt, lengths = preprocess(data) 

    # inference
    session = onnxruntime.InferenceSession("crnn.onnx")
    xt = xt.to('cpu').detach().numpy().copy()
    lengths = lengths.to('cpu').detach().numpy().copy()
    results = session.run(["conf"],{ "data": xt, "lengths": lengths})

    x = torch.from_numpy(results[0].astype(np.float32)).clone()

    label, conf = postprocess(x)

    return label, conf

def main():
    path = "dog.wav" #dog_bark 0.5050086379051208
    label, conf = crnn(path)
    print(label)
    print(conf)


def main_ailia():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    #env_id = ailia.get_gpu_environment_id()
    #print(f'env_id: {env_id}')
    #net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(input_data)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end-start))
    else:
        preds_ailia = net.predict(input_data)

    # masked word prediction
    predicted_indices = np.argsort(
        preds_ailia[0][0][masked_index]
    )[-NUM_PREDICT:][::-1]

    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

    print('Input sentence: ' + SENTENCE)
    print(f'predicted top {NUM_PREDICT} words: {predicted_tokens}')
    print('Script finished successfully.')


if __name__ == "__main__":
    main()
