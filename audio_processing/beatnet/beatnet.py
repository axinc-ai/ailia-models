import scipy
import pyaudio
import numpy as np
from madmom.features import DBNDownBeatTrackingProcessor
from particle_filtering_cascade import particle_filter_cascade



import sys
import time

import ailia

import numpy as np

# import original modules
sys.path.append('../../util')

# logger
from logging import getLogger  # noqa: E402

from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402

import matplotlib.pyplot as plt

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

MODEL_NAME = "beatnet"

MODEL_PATH = MODEL_NAME + '.onnx.prototxt'

WEIGHT_PATH_1 = MODEL_NAME + "_1.onnx"
WEIGHT_PATH_2 = MODEL_NAME + "_2.onnx"
WEIGHT_PATH_3 = MODEL_NAME + "_3.onnx"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/" + MODEL_NAME + "/"

DEFAULT_INPUT_PATH = 'input.mp3'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking',
    DEFAULT_INPUT_PATH, None
)

parser.add_argument('--mode', type=str, default='offline', help='Working mode: realtime, online, or offline')

parser.add_argument('--inference-model', type=str, default='DBN', help='Inference model to use: PF (Particle Filtering) or DBN (Dynamic Bayesian Network).')

parser.add_argument('--weights', type=str, default='1', help='Model weights to use (1, 2, or 3). 1 is trained with GTZAN, 2 is trained with Ballroom, 3 is trained with Rock_corpus. Default is 1')

args = update_parser(parser)

# ======================
# Helper functions
# ======================

# From https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/log_spect.py
# Imports were modified to not require BeatNet(pip module)

# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>

from beatnet_common import *

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor


# feature extractor that extracts magnitude spectrogram and its differences  

class LOG_SPECT(FeatureModule):
    def __init__(self, num_channels=1, sample_rate=22050, win_length=2048, hop_size=512, n_bands=[12], mode='online'):
        sig = SignalProcessor(num_channels=num_channels, win_length=win_length, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        multi = ParallelProcessor([])
        frame_sizes = [win_length]  
        num_bands = n_bands  
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            if mode == 'online' or mode == 'offline':
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size) 
            else:   # for real-time and streaming modes 
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, num_frames=4) 
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and process everything sequentially
        self.pipe = SequentialProcessor((sig, multi, np.hstack))

    def process_audio(self, audio):
        feats = self.pipe(audio)
        return feats.T

# particle filter cascade
# From 

# ======================
# Main functions
# ======================
'''
From https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/log_spect.py
__init__, activation_extractor_stream, activation_extractor_realtime, activation_extractor_online was modified to run ailia models instead of PyTorch models
Removed the following features:
- threading
- device

'''
class BeatNet:

    '''
    The main BeatNet handler class including different trained models, different modes for extracting the activation and causal and non-causal inferences

        Parameters
        ----------
        Inputs: 
            model: A scalar in the range [1,3] to select which pre-trained CRNN models to utilize. 
            mode: A string to determine the working mode. i.e. 'stream', 'realtime', 'online', and ''offline.
                'stream' mode: Uses the system microphone to capture sound and does the process in real time. Due to training the model on standard mastered songs, it is highly recommended to make sure the microphone sound is as loud as possible. Less reverbrations leads to the better results.  
                'Realtime' mode: Reads an audio file chunk by chunk, and processes each chunk at the time.
                'Online' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then infers the parameters on interest using particle filtering.
                'offline' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then inferes the parameters on interest using madmom dynamic Bayesian network. This method is quicker that madmom beat/downbeat tracking.
            inference model: A string to choose the inference approach. i.e. 'PF' standing for Particle Filtering for causal inferences and 'DBN' standing for Dynamic Bayesian Network for non-causal usages.
            plot: A list of strings to plot. 
                'activations': Plots the neural network activations for beats and downbeats of each time frame. 
                'beat_particles': Plots beat/tempo tracking state space and current particle states at each time frame.
                'downbeat_particles': Plots the downbeat/meter tracking state space and current particle states at each time frame.
                Note that to speedup plotting the figures, rather than new plots per frame, the previous plots get updated. However, to secure realtime results, it is recommended to not plot or have as less number of plots as possible at the time.   
            threading: To decide whether to accomplish the inference at the main thread or another thread. 
            device: type of dvice. cpu or cuda:i

        Outputs:
            A vector including beat times and downbeat identifier columns, respectively with the following shape: numpy_array(num_beats, 2).
    '''
    
    
    def __init__(self, model, mode='online', inference_model='PF', plot=['beat_particles']):
        self.model = model
        self.mode = mode
        self.inference_model = inference_model
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)
        if self.inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=plot, mode=self.mode)
        elif self.inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')

        self.model = model
                                             
    def process(self, audio_path=None):
        if self.mode == "realtime":
            self.counter = 0
            self.completed = 0
            if self.inference_model != "PF":
                raise RuntimeError('The infernece model for the streaming mode should be set to "PF".')
            if isinstance(audio_path, str) or audio_path.all()!=None:
                while self.completed == 0:
                    self.activation_extractor_realtime(audio_path) # Using BeatNet causal Neural network realtime mode to extract activations
                    output = self.estimator.process(self.pred)  # Using particle filtering online inference to infer beat/downbeats
                    self.counter += 1
                return output
            else:
                raise RuntimeError('An audio object or file directory is required for the realtime usage!')
        
        
        elif self.mode == "online":
            if isinstance(audio_path, str) or audio_path.all()!=None:
                preds = self.activation_extractor_online(audio_path)    # Using BeatNet causal Neural network to extract activations
            else:
                raise RuntimeError('An audio object or file directory is required for the online usage!')
            if self.inference_model == "PF":   # Particle filtering inference (causal)
                output = self.estimator.process(preds)  # Using particle filtering online inference to infer beat/downbeats
                return output
            elif self.inference_model == "DBN":    # Dynamic bayesian Network Inference (non-causal)
                output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                return output
        
        
        elif self.mode == "offline":
                if self.inference_model != "DBN":
                    raise RuntimeError('The infernece model should be set to "DBN" for the offline mode!')
                if isinstance(audio_path, str) or audio_path.all()!=None:
                    preds = self.activation_extractor_online(audio_path)    # Using BeatNet causal Neural network to extract activations
                    output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                    return output
        
                else:
                    raise RuntimeError('An audio object or file directory is required for the offline usage!')


    def activation_extractor_realtime(self, audio_path):
        if self.counter==0: #loading the audio
            self.audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
        if self.counter<(round(len(self.audio)/self.log_spec_hop_length)):
            if self.counter<2:
                self.pred = np.zeros([1,2])
            else:
                feats = self.proc.process_audio(self.audio[self.log_spec_hop_length * (self.counter-2):self.log_spec_hop_length * (self.counter) + self.log_spec_win_length]).T[-1]
                feats = feats[None,None]
                pred = self.model.predict(feats)[0]
                pred = scipy.special.softmax(pred, axis=0)
                self.pred = np.transpose(pred[:2, :])
        else:
            self.completed = 1


    def activation_extractor_online(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=22050)  # reading the data
        feats = self.proc.process_audio(audio).T
        feats = feats[None]
        preds = self.model.predict(feats) # extracting the activations by passing the feature through the NN
        preds = preds[0]

        preds = scipy.special.softmax(preds, axis=0)
        preds = np.transpose(preds[:2, :])
        return preds

def predict(model):
    logger.info('Start inference...')
    beatnet = BeatNet(model, mode=args.mode, inference_model=args.inference_model, plot=[''])

    audio_path = args.input[0]

    if args.benchmark and not (args.video is not None):
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            out = beatnet.process(audio_path)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        out = beatnet.process(audio_path)

    logger.info('Predicted beats:')
    logger.info(out)

    logger.info('Script finished successfully.')

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_1, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_2, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_3, MODEL_PATH, REMOTE_PATH)
    
    # net initialize
    WEIGHT_PATH = [WEIGHT_PATH_1, WEIGHT_PATH_2, WEIGHT_PATH_3][int(args.weights) - 1]
    model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id = args.env_id)

    predict(model)


if __name__ == '__main__':
    main()
