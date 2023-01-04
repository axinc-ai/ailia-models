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

from speechsplit_utils import pad_seq_to_2, quantize_f0_numpy


# ======================
# PARAMETERS
# ======================
WAV_PATH = 'input.wav'
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
    'Unsupervised Speech Decomposition Via Triple Information Bottleneck', WAV_PATH, SAVE_WAV_PATH
)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='WAV', default=WAV_PATH,
    help='input text'
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Main function
# ======================
def audio_recognition(net_converter, net_generator):
    import torch
    import pickle
    metadata = pickle.load(open('demo.pkl', "rb"))

    sbmt_i = metadata[0]
    emb_org = torch.from_numpy(sbmt_i[1])
    x_org, f0_org, len_org, uid_org = sbmt_i[2]        
    uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis,:,:], 192)
    uttr_org_pad = torch.from_numpy(uttr_org_pad)
    f0_org_pad = np.pad(f0_org, (0, 192-len_org), 'constant', constant_values=(0, 0))
    f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
    f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
    f0_org_onehot = torch.from_numpy(f0_org_onehot)
    uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

    sbmt_j = metadata[1]
    emb_trg = torch.from_numpy(sbmt_j[1])
    x_trg, f0_trg, len_trg, uid_trg = sbmt_j[2]        
    uttr_trg_pad, len_trg_pad = pad_seq_to_2(x_trg[np.newaxis,:,:], 192)
    uttr_trg_pad = torch.from_numpy(uttr_trg_pad)
    f0_trg_pad = np.pad(f0_trg, (0, 192-len_trg), 'constant', constant_values=(0, 0))
    f0_trg_quantized = quantize_f0_numpy(f0_trg_pad)[0]
    f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
    f0_trg_onehot = torch.from_numpy(f0_trg_onehot)

    # f0_converter
    uttr_org_pad = uttr_org_pad.to('cpu').detach().numpy().copy()
    f0_trg_onehot = f0_trg_onehot.to('cpu').detach().numpy().copy()
    f0_pred = net_converter.predict([uttr_org_pad, f0_trg_onehot])[0]
    uttr_org_pad = torch.from_numpy(uttr_org_pad)
    f0_trg_onehot = torch.from_numpy(f0_trg_onehot)
    f0_pred = torch.from_numpy(f0_pred)
    f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    f0_con_onehot = torch.zeros((1, 192, 257))
    f0_con_onehot[0, torch.arange(192), f0_pred_quantized] = 1
    # end of f0_converter

    uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1) 

    # generator
    conditions = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']
    spect_vc = []
    uttr_f0_org = uttr_f0_org.to('cpu').detach().numpy().copy()
    uttr_f0_trg = uttr_f0_trg.to('cpu').detach().numpy().copy()
    uttr_org_pad = uttr_org_pad.to('cpu').detach().numpy().copy()
    uttr_trg_pad = uttr_trg_pad.to('cpu').detach().numpy().copy()
    emb_org = emb_org.to('cpu').detach().numpy().copy()
    emb_trg = emb_trg.to('cpu').detach().numpy().copy()
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
    # end of generator

    # spectrogram to waveform
    use_original_wavenet_vocoder = True

    if use_original_wavenet_vocoder:
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
