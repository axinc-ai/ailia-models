import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import torch
from scipy import signal
from pysptk import sptk
import soundfile as sf
from librosa.filters import mel
from speechsplit_utils import (
    pad_seq_to_2, 
    quantize_f0_numpy,
    butter_highpass,
    pySTFT,
    speaker_normalization
)


# ======================
# PARAMETERS
# ======================
WAV_PATH_ORG = 'input_org.wav'
WAV_PATH_TRG = 'input_trg.wav'
SAVE_WAV_PATH = 'output.wav'

WEIGHT_PATH_CONVERTER = 'F0_Converter.onnx'
MODEL_PATH_CONVERTER = 'F0_Converter.onnx.prototxt'
WEIGHT_PATH_GENERATOR = 'Generator.onnx'
MODEL_PATH_GENERATOR = 'Generator.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/speechsplit/'


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Unsupervised Speech Decomposition Via Triple Information Bottleneck', WAV_PATH_ORG, SAVE_WAV_PATH
)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='WAV', default=WAV_PATH_ORG,
    help='input audio'
)
parser.add_argument(
    '--input2', '-i2', metavar='WAV', default=WAV_PATH_TRG,
    help='input2 audio'
)
parser.add_argument(
    '--input_gender', '-g', type=str, default='M',
    choices=['M', 'F'],
    help='gender of person who speaks in input audio.'
)
parser.add_argument(
    '--input2_gender', '-g2', type=str, default='F',
    choices=['M', 'F'],
    help='gender of person who speaks in input2 audio.'
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
parser.add_argument(
    '--use_ailia_wavenet_vocoder', action='store_true',
    help='use wavenet_vocoder added to ailia library'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main function
# ======================
def make_spect_f0(wav, mf):
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    if mf == 'M':
        lo, hi = 50, 250
    elif mf == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError

    # read audio file
    x, fs = sf.read(wav)
    assert fs == 16000
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (np.random.rand(y.shape[0])-0.5)*1e-06

    # compute spectrogram
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100

    # extract f0
    f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    assert len(S) == len(f0_rapt)

    return S.astype(np.float32), f0_norm.astype(np.float32)


def get_speaker_embbed(gender):
    # embbed 1 as a temporary measure because this model needs speaker's id in trained dataset.
    # trained dataset has 82 person's voice.
    # we have to do fine-turning in order to eliminate speaker's id problem in future.
    if gender == 'M':
        emb = np.zeros((1, 82), dtype=np.float32)
        emb[0, 1] = 1
    elif gender == 'F':
        emb = np.zeros((1, 82), dtype=np.float32)
        emb[0, 6] = 1
    else:
        print('Invalid argument.')
        exit()

    return emb


def audio_recognition(net_converter, net_generator):
    input_file = args.input[0] # TODO: loop

    # create input data
    S_org, f0_norm_org = make_spect_f0(input_file, args.input_gender)
    S_org = S_org[:192, :]
    f0_norm_org = f0_norm_org[:192]
    emb_org = get_speaker_embbed(args.input_gender)
    sbmt_org_2 = [S_org, f0_norm_org, len(f0_norm_org), None]

    S_trg, f0_norm_trg = make_spect_f0(args.input2, args.input2_gender)
    S_trg = S_trg[:192, :]
    f0_norm_trg = f0_norm_trg[:192]
    emb_trg = get_speaker_embbed(args.input2_gender)
    sbmt_trg_2 = [S_trg, f0_norm_trg, len(f0_norm_trg), None]

    metadata = [
        [None, emb_org, sbmt_org_2], 
        [None, emb_trg, sbmt_trg_2]
    ]

    # preprocess
    sbmt_i = metadata[0]
    emb_org = sbmt_i[1]
    x_org, f0_org, len_org, uid_org = sbmt_i[2] 
    uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis,:,:], 192)
    uttr_org_pad = torch.from_numpy(uttr_org_pad)
    f0_org_pad = np.pad(f0_org, (0, 192-len_org), 'constant', constant_values=(0, 0))
    f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
    f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
    f0_org_onehot = torch.from_numpy(f0_org_onehot)
    uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

    sbmt_j = metadata[1]
    emb_trg = sbmt_j[1]
    x_trg, f0_trg, len_trg, uid_trg = sbmt_j[2]        
    uttr_trg_pad, len_trg_pad = pad_seq_to_2(x_trg[np.newaxis,:,:], 192)
    uttr_trg_pad = torch.from_numpy(uttr_trg_pad)
    f0_trg_pad = np.pad(f0_trg, (0, 192-len_trg), 'constant', constant_values=(0, 0))
    f0_trg_quantized = quantize_f0_numpy(f0_trg_pad)[0]
    f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
    f0_trg_onehot = torch.from_numpy(f0_trg_onehot)

    ### START OF f0_converter ###
    uttr_org_pad = uttr_org_pad.to('cpu').detach().numpy().copy()
    f0_trg_onehot = f0_trg_onehot.to('cpu').detach().numpy().copy()
    f0_pred = net_converter.predict([uttr_org_pad, f0_trg_onehot])[0]
    uttr_org_pad = torch.from_numpy(uttr_org_pad)
    f0_trg_onehot = torch.from_numpy(f0_trg_onehot)
    f0_pred = torch.from_numpy(f0_pred)
    f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    f0_con_onehot = torch.zeros((1, 192, 257))
    f0_con_onehot[0, torch.arange(192), f0_pred_quantized] = 1
    ### END OF f0_converter ###

    uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1) 

    ### START OF generator ###
    conditions = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']
    spect_vc = []
    uttr_f0_org = uttr_f0_org.to('cpu').detach().numpy().copy()
    uttr_f0_trg = uttr_f0_trg.to('cpu').detach().numpy().copy()
    uttr_org_pad = uttr_org_pad.to('cpu').detach().numpy().copy()
    uttr_trg_pad = uttr_trg_pad.to('cpu').detach().numpy().copy()

    for condition in conditions:
        if condition == 'R':
            x_identic_val = net_generator.predict([uttr_f0_org, uttr_trg_pad, emb_org])[0]
        if condition == 'F':
            x_identic_val = net_generator.predict([uttr_f0_trg, uttr_org_pad, emb_org])[0]
        if condition == 'U':
            x_identic_val = net_generator.predict([uttr_f0_org, uttr_org_pad, emb_trg])[0]
        if condition == 'RF':
            x_identic_val = net_generator.predict([uttr_f0_trg, uttr_trg_pad, emb_org])[0]
        if condition == 'RU':
            x_identic_val = net_generator.predict([uttr_f0_org, uttr_trg_pad, emb_trg])[0]
        if condition == 'FU':
            x_identic_val = net_generator.predict([uttr_f0_trg, uttr_org_pad, emb_trg])[0]
        if condition == 'RFU':
            x_identic_val = net_generator.predict([uttr_f0_trg, uttr_trg_pad, emb_trg])[0]

        if 'R' in condition:
            uttr_trg = x_identic_val[0, :len_trg, :]
        else:
            uttr_trg = x_identic_val[0, :len_org, :]
                
        spect_vc.append( ('{}_{}_{}_{}'.format(sbmt_i[0], sbmt_j[0], uid_org, condition), uttr_trg ) ) 
    ### END OF generator ###

    # spectrogram to waveform
    if not args.use_ailia_wavenet_vocoder:
        import os
        import soundfile
        from synthesis import build_model
        from synthesis import wavegen

        if not os.path.exists('results'):
            os.makedirs('results')

        model = build_model()
        checkpoint = torch.load("checkpoint_step001000000_ema.pth", map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])

        for spect in spect_vc:
            name = spect[0]
            c = spect[1]
            print(name)
            print(c.shape)
            waveform = wavegen(model, c=c)   
            soundfile.write('results/'+name+'.wav', waveform, samplerate=16000)
    else:
        print('Not implemented.')

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH_CONVERTER, MODEL_PATH_CONVERTER, REMOTE_PATH)
    #check_and_download_models(WEIGHT_PATH_GENERATOR, MODEL_PATH_GENERATOR, REMOTE_PATH)

    env_id = args.env_id

    net_converter = ailia.Net(MODEL_PATH_CONVERTER, WEIGHT_PATH_CONVERTER, env_id=env_id)
    net_generator = ailia.Net(MODEL_PATH_GENERATOR, WEIGHT_PATH_GENERATOR, env_id=env_id)

    audio_recognition(net_converter, net_generator)


if __name__ == "__main__":
    main()
