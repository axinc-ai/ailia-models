import sys
import time
from logging import getLogger

import numpy as np
import scipy.signal as signal
from PIL import Image
import librosa
import soundfile as sf

import ailia

# import original modules
sys.path.append('../../util')
from microphone_utils import start_microphone_input  # noqa
from model_utils import check_and_download_models  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

flg_ffmpeg = False

if flg_ffmpeg:
    import ffmpeg

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DECODER_PATH = "ar_decoder.opt.onnx"
MODEL_DECODER_PATH = "ar_decoder.opt.onnx.prototxt"
WEIGHT_ENCODEC_PATH = "encodec.onnx"
MODEL_ENCODEC_PATH = "encodec.onnx.prototxt"
WEIGHT_VOCOS_PATH = "vocos.opt.onnx"
MODEL_VOCOS_PATH = "vocos.opt.onnx.prototxt"
WEIGHT_AUDIO_EMBEDDING_PATH = "audio_embedding.onnx"
MODEL_AUDIO_EMBEDDING_PATH = "audio_embedding.onnx.prototxt"

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/vall-e-x/'

SAMPLE_RATE = 16000

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'VALL-E-X', None, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default=None,
    help='input text'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='use onnx runtime'
)
parser.add_argument(
    '--profile', action='store_true',
    help='use profile model'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Parameters
# ======================

if args.onnx:
    import onnxruntime
else:
    import ailia

if args.input:
    text = args.input
else:
    text ="こんにちは。今日は新しいAIエンジンであるアイリアSDKを紹介します。アイリアSDKは高速なAI推論エンジンです。"

sampling_rate = 22050

def generate_voice(decoder, encodec, audio_embedding, vocos):
    # onnx
    logger.info("Input text : " + text)

    texts = [text]

    #mel_outputs_postnet = test_inference(texts, encoder, decoder_iter, postnet)

    #stride = 256 # value from waveglow upsample
    #n_group = 8
    #z_size2 = (mel_outputs_postnet.shape[2]*stride)//n_group
    #z = np.random.randn(1, n_group, z_size2).astype(np.float32)

    #print("Running Tacotron2 Waveglow")

    #if args.benchmark:
    #    start = int(round(time.time() * 1000))

    #if args.onnx:
    #    waveglow_inputs = {waveglow.get_inputs()[0].name: mel_outputs_postnet,
    #                    waveglow.get_inputs()[1].name: z}
    #    audio = waveglow.run(None, waveglow_inputs)[0]
    #else:
    #    waveglow_inputs = [mel_outputs_postnet, z]
    #    audio = waveglow.run(waveglow_inputs)[0]

    #if args.benchmark:
    #    end = int(round(time.time() * 1000))
    #    estimation_time = (end - start)
    #    logger.info(f'\twavegrow processing time {estimation_time} ms')

    # export to audio
    #savepath = args.savepath
    #logger.info(f'saved at : {savepath}')
    #sf.write(savepath, audio[0].astype(np.float32), sampling_rate)
    #logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_DECODER_PATH, MODEL_DECODER_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENCODEC_PATH, MODEL_ENCODEC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_AUDIO_EMBEDDING_PATH, MODEL_AUDIO_EMBEDDING_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_VOCOS_PATH, MODEL_VOCOS_PATH, REMOTE_PATH)

    #env_id = args.env_id

    if args.onnx:
        decoder = onnxruntime.InferenceSession(WEIGHT_DECODER_PATH)
        encodec = onnxruntime.InferenceSession(WEIGHT_ENCODEC_PATH)
        audio_embedding = onnxruntime.InferenceSession(WEIGHT_AUDIO_EMBEDDING_PATH)
        vocos = onnxruntime.InferenceSession(WEIGHT_VOCOS_PATH)
    else:
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        decoder = ailia.Net(stream = MODEL_DECODER_PATH, weight = WEIGHT_DECODER_PATH, memory_mode = memory_mode, env_id = args.env_id)
        encodec = ailia.Net(stream = MODEL_ENCODEC_PATH, weight = WEIGHT_ENCODEC_PATH, memory_mode = memory_mode, env_id = args.env_id)
        audio_embedding = ailia.Net(stream = MODEL_AUDIO_EMBEDDING_PATH, weight = WEIGHT_AUDIO_EMBEDDING_PATH, memory_mode = memory_mode, env_id = args.env_id)
        vocos = ailia.Net(stream = MODEL_VOCOS_PATH, weight = WEIGHT_VOCOS_PATH, memory_mode = memory_mode, env_id = args.env_id)
        if args.profile:
            decoder.set_profile_mode(True)
            encodec.set_profile_mode(True)
            audio_embedding.set_profile_mode(True)
            vocos.set_profile_mode(True)

    generate_voice(decoder, encodec, audio_embedding, vocos)

    if args.profile:
        print("decoder : ")
        print(decoder.get_summary())
        print("encodec : ")
        print(encodec.get_summary())
        print("audio_embedding : ")
        print(audio_embedding.get_summary())
        print("vocos : ")
        print(vocos.get_summary())

if __name__ == '__main__':
    main()
