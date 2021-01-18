import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402
import onnxruntime

import soundfile as sf
from scipy import signal

# import original modules
sys.path.append('../util')
from model_utils import check_and_download_models  # noqa: E402
from unet_source_separation_utils import preemphasis, inv_preemphasis, lowpass, tfconvert, zero_pad, calc_time  # noqa: E402


# ======================
# Parameters 1
# ======================
WAV_PATH = 'doublenoble_k7rain_part.wav' # noisy speech sample
#WAVE_PATH = '049 - Young Griffo - Facade.wav' # music sample
SAVE_WAV_PATH = 'separated_voice.wav'  
MODEL_LISTS = ['base', 'large']


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Source separation.'
)
parser.add_argument(
    '-i', '--input', metavar='WAVFILE',
    default=WAV_PATH,
    help='The input audio path.'
)
parser.add_argument(
    '-o', '--savepath',
    default=SAVE_WAV_PATH,
    help='The output audio path.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. '
)
parser.add_argument(
    '-n', '--onnx', 
    action='store_true',
    default=True,
    help='Use onnxruntime'
)
parser.add_argument(
    '-a', '--arch', 
    default='base', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = parser.parse_args()


# ======================
# Parameters 2
# ======================

if args.arch == 'base' : # for general voice separation
    WEIGHT_PATH = "second_voice_bank.best.opt.onnx"
else :  # for singing voice separation
    WEIGHT_PATH = "RefineSpectrogramUnet.best.opt.onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/unet_source_separation/"

# fixed parameters for each model
if args.arch == 'base' :
    DESIRED_SR = 22050
    MULT = 2 ** 5
    WINDOW_LEN = 512
    HOP_LEN = 64
else :
    DESIRED_SR = 44100
    MULT = 2 ** 6
    WINDOW_LEN = 1024
    HOP_LEN = 128 

# adjustable parameters 
if args.arch == 'base' :
    LPF_CUTOFF = 10000
else :
    LPF_CUTOFF = 20000


# ======================
# Main function
# ======================
def src_sep(data, session) :
    # inference
    if not args.onnx :
        sep = session.predict(data)

    else :
        first_input_name = session.get_inputs()[0].name
        second_input_name = session.get_inputs()[1].name
        first_output_name = session.get_outputs()[0].name
        sep = session.run(
            [first_output_name], 
            {first_input_name: data[0], second_input_name: data[1]})[0]

    return sep


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load audio
    print('Loading wavfile...')
    wav, sr = sf.read(args.input)
    
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    if wav.ndim == 2 :
        wav = wav[:,0][np.newaxis,:]
    else:
        wav = wav[np.newaxis,:]

    calc_time(wav.shape[1], sr)

    # convert sample rate
    print('Converting sample rate...')
    if not sr == DESIRED_SR :
        wav = signal.resample_poly(wav, DESIRED_SR, sr, axis=1)

    # apply preenphasis filter
    print('Generating input feature...')
    wav = preemphasis(wav)

    input_feature = tfconvert(wav, WINDOW_LEN, HOP_LEN, MULT)

    # create instance
    if not args.onnx :
        print('Use ailia')
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else :
        print('Use onnxruntime')
        session = onnxruntime.InferenceSession(WEIGHT_PATH)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for c in range(5) :
            start = int(round(time.time() * 1000))
            sep = src_sep(input_feature, session)
            end = int(round(time.time() * 1000))
            print("\tprocessing time {} ms".format(end-start))
    else:
        sep = src_sep(input_feature, session)

    # postprocessing
    print('Start postprocessing...')
    if LPF_CUTOFF > 0 :
        sep = lowpass(sep, LPF_CUTOFF, DESIRED_SR)

    out_wav = inv_preemphasis(sep).clip(-1.,1.)
    out_wav = out_wav.swapaxes(0,1)
    
    # save sapareted signal
    sf.write(args.savepath, out_wav, DESIRED_SR)
    
    print('Saved separated signal. ')
    print('Script finished successfully.')


if __name__ == "__main__":
     main()
