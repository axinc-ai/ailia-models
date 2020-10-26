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
from crnn_audio_classification_util import MelspectrogramStretch, SpecNormalization, AudioTransforms, ProcessChannels, ToTensorAudio  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================

WAVE_PATH="24965__www-bonson-ca__bigdogbarking-02.wav" #https://freesound.org/people/www.bonson.ca/sounds/24965/
#WAVE_PATH="dog.wav" #dog_bark 0.5050086379051208

parser = argparse.ArgumentParser(
    description='CRNN Audio Classification.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=WAVE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-o', '--onnx',
    action='store_true',
    help='Running on onnx runtime'
)

args = parser.parse_args()


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "crnn_audio_classification.onnx"
MODEL_PATH = "crnn_audio_classification.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crnn_audio_classification/"

# ======================
# Preprocess
# ======================


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

def crnn(path, session):
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
    xt = xt.to('cpu').detach().numpy().copy()
    lengths = lengths.to('cpu').detach().numpy().copy()
    if args.onnx:
        results = session.run(["conf"],{ "data": xt, "lengths": lengths})
    else:
        results = net.predict({ "data": xt, "lengths": lengths})

    x = torch.from_numpy(results[0].astype(np.float32)).clone()

    label, conf = postprocess(x)

    return label, conf

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    path = args.input
    if args.onnx:
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            label, conf = crnn(path, session)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end-start))
    else:
        label, conf = crnn(path, session)

    print(label)
    print(conf)

    print('Script finished successfully.')


if __name__ == "__main__":
    main()
