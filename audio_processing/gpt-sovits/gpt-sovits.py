import time
import sys
import argparse
import re

import numpy as np
import soundfile as sf

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from scipy.io.wavfile import write
# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import os
from text import cleaned_text_to_sequence
from text.japanese import g2p
import soundfile
import librosa

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gpt-sovits/'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'GPT-SoVits', None, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    '--input', '-i', metavar='TEXT', default="ax株式会社ではAIの実用化のための技術を開発しています。",
    help='input text'
)
parser.add_argument(
    '--audio', '-a', metavar='TEXT', default="reference_audio_captured_by_ax.wav",
    help='ref audio'
)
parser.add_argument(
    '--transcript', '-t', metavar='TEXT', default="水をマレーシアから買わなくてはならない。",
    help='ref text'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='use onnx runtime'
)
parser.add_argument(
    '--onnx_vits', action='store_true',
    help='use onnx runtime for vits'
)
parser.add_argument(
    '--profile', action='store_true',
    help='use profile model'
)
args = update_parser(parser, check_input_type=False)

WEIGHT_PATH_SSL = 'nahida_cnhubert.onnx'
WEIGHT_PATH_T2S_ENCODER = 'nahida_t2s_encoder.onnx'
WEIGHT_PATH_T2S_FIRST_DECODER = 'nahida_t2s_fsdec.onnx'
WEIGHT_PATH_T2S_STAGE_DECODER = 'nahida_t2s_sdec.onnx'
WEIGHT_PATH_VITS = 'nahida_vits.onnx'

MODEL_PATH_SSL = 'nahida_cnhubert.onnx.prototxt'
MODEL_PATH_T2S_ENCODER = 'nahida_t2s_encoder.onnx.prototxt'
MODEL_PATH_T2S_FIRST_DECODER = 'nahida_t2s_fsdec.onnx.prototxt'
MODEL_PATH_T2S_STAGE_DECODER = 'nahida_t2s_sdec.onnx.prototxt'
MODEL_PATH_VITS = 'nahida_vits.onnx.prototxt'


# ======================
# Logic
# ======================

class T2SModel():
    def __init__(self, sess_encoder, sess_fsdec, sess_sdec):
        self.hz = 50
        self.max_sec = 54
        self.top_k = 5
        self.early_stop_num = np.array([self.hz * self.max_sec])
        self.sess_encoder = sess_encoder
        self.sess_fsdec = sess_fsdec
        self.sess_sdec = sess_sdec

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.early_stop_num

        top_k = np.array([5], dtype=np.int64)
        top_p = np.array([1.0], dtype=np.float32)
        temperature = np.array([1.0], dtype=np.float32)
        repetition_penalty = np.array([1.35], dtype=np.float32)

        EOS = 1024

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            x, prompts = self.sess_encoder.run(None, {"ref_seq":ref_seq, "text_seq":text_seq, "ref_bert":ref_bert, "text_bert":text_bert, "ssl_content":ssl_content})
        else:
            x, prompts = self.sess_encoder.run({"ref_seq":ref_seq, "text_seq":text_seq, "ref_bert":ref_bert, "text_bert":text_bert, "ssl_content":ssl_content})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tsencoder processing time {} ms".format(end-start))

        prefix_len = prompts.shape[1]

        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            y, k, v, y_emb, x_example = self.sess_fsdec.run(None, {"x":x, "prompts":prompts, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
        else:
            y, k, v, y_emb, x_example = self.sess_fsdec.run({"x":x, "prompts":prompts, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tfsdec processing time {} ms".format(end-start))

        stop = False
        for idx in range(1, 1500):
            if args.benchmark:
                start = int(round(time.time() * 1000))
            if args.onnx:
                y, k, v, y_emb, logits, samples = self.sess_sdec.run(None, {"iy":y, "ik":k, "iv":v, "iy_emb":y_emb, "ix_example":x_example, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
            else:
                y, k, v, y_emb, logits, samples = self.sess_sdec.run({"iy":y, "ik":k, "iv":v, "iy_emb":y_emb, "ix_example":x_example, "top_k":top_k, "top_p":top_p, "temperature":temperature, "repetition_penalty":repetition_penalty})
            if args.benchmark:
                end = int(round(time.time() * 1000))
                logger.info("\tsdec processing time {} ms".format(end-start))
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if np.argmax(logits, axis=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[np.newaxis, :, -idx:-1]


class GptSoVits():
    def __init__(self, t2s, sess):
        self.t2s = t2s
        self.sess = sess
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s.forward(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx or args.onnx_vits:
            audio1 = self.sess.run(None, {
                "text_seq" : text_seq,
                "pred_semantic" : pred_semantic, 
                "ref_audio" : ref_audio
            })
        else:
            audio1 = self.sess.run({
                "text_seq" : text_seq,
                "pred_semantic" : pred_semantic, 
                "ref_audio" : ref_audio
            })
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tvits processing time {} ms".format(end-start))
        return audio1[0]


class SSLModel():
    def __init__(self, sess):
        self.sess = sess

    def forward(self, ref_audio_16k):
        if args.benchmark:
            start = int(round(time.time() * 1000))
        if args.onnx:
            last_hidden_state = self.sess.run(None, {
                "ref_audio_16k" : ref_audio_16k
            })
        else:
            last_hidden_state = self.sess.run({
                "ref_audio_16k" : ref_audio_16k
            })
        if args.benchmark:
            end = int(round(time.time() * 1000))
            logger.info("\tssl processing time {} ms".format(end-start))
        return last_hidden_state[0]


def generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits):
    gpt = T2SModel(t2s_encoder, t2s_first_decoder, t2s_stage_decoder,)
    gpt_sovits = GptSoVits(gpt, vits)
    ssl = SSLModel(ssl)

    input_audio = args.audio

    ref_phones = g2p(args.transcript)
    ref_seq = np.array([cleaned_text_to_sequence(ref_phones)], dtype=np.int64)

    text_phones = g2p(args.input)
    text_seq = np.array([cleaned_text_to_sequence(text_phones)], dtype=np.int64)

    # empty for ja or en
    ref_bert = np.zeros((ref_seq.shape[1], 1024), dtype=np.float32)
    text_bert = np.zeros((text_seq.shape[1], 1024), dtype=np.float32)
    
    vits_hps_data_sampling_rate = 32000

    zero_wav = np.zeros(
        int(vits_hps_data_sampling_rate * 0.3),
        dtype=np.float32,
    )
    wav16k, sr = librosa.load(input_audio, sr=16000)
    wav16k = np.concatenate([wav16k, zero_wav], axis=0)
    wav16k = wav16k[np.newaxis, :]
    ref_audio_16k = wav16k # hubertの入力のみpaddingする

    wav32k, sr = librosa.load(input_audio, sr=vits_hps_data_sampling_rate)
    wav32k = wav32k[np.newaxis, :]

    ssl_content = ssl.forward(ref_audio_16k)

    a = gpt_sovits.forward(ref_seq, text_seq, ref_bert, text_bert, wav32k, ssl_content)

    savepath = args.savepath
    logger.info(f'saved at : {savepath}')

    soundfile.write(savepath, a, vits_hps_data_sampling_rate)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_SSL, MODEL_PATH_SSL, WEIGHT_PATH_SSL)
    check_and_download_models(WEIGHT_PATH_T2S_ENCODER, MODEL_PATH_T2S_ENCODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_T2S_FIRST_DECODER, MODEL_PATH_T2S_FIRST_DECODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_T2S_STAGE_DECODER, MODEL_PATH_T2S_STAGE_DECODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_VITS, MODEL_PATH_VITS, REMOTE_PATH)

    #env_id = args.env_id

    if args.onnx:
        import onnxruntime
        ssl = onnxruntime.InferenceSession(WEIGHT_PATH_SSL)
        t2s_encoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_ENCODER)
        t2s_first_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_FIRST_DECODER)
        t2s_stage_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_STAGE_DECODER)
        vits = onnxruntime.InferenceSession(WEIGHT_PATH_VITS)
    else:
        import ailia
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        ssl = ailia.Net(weight = WEIGHT_PATH_SSL, stream = MODEL_PATH_SSL, memory_mode = memory_mode, env_id = args.env_id)
        t2s_encoder = ailia.Net(weight = WEIGHT_PATH_T2S_ENCODER, stream = MODEL_PATH_T2S_ENCODER, memory_mode = memory_mode, env_id = args.env_id)
        t2s_first_decoder = ailia.Net(weight = WEIGHT_PATH_T2S_FIRST_DECODER, stream = MODEL_PATH_T2S_FIRST_DECODER, memory_mode = memory_mode, env_id = args.env_id)
        t2s_stage_decoder = ailia.Net(weight = WEIGHT_PATH_T2S_STAGE_DECODER, stream = MODEL_PATH_T2S_STAGE_DECODER, memory_mode = memory_mode, env_id = args.env_id)
        if args.onnx_vits:
            import onnxruntime
            vits = onnxruntime.InferenceSession(WEIGHT_PATH_VITS) # stftの対応までの暫定
        else:
            vits = ailia.Net(weight = WEIGHT_PATH_VITS, stream = MODEL_PATH_VITS, memory_mode = memory_mode, env_id = args.env_id)
        if args.profile:
            ssl.set_profile_mode(True)
            t2s_encoder.set_profile_mode(True)
            t2s_first_decoder.set_profile_mode(True)
            t2s_stage_decoder.set_profile_mode(True)
            if not args.onnx_vits:
                vits.set_profile_mode(True)

    generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits)

    if args.profile:
        print("ssl : ")
        print(ssl.get_summary())
        print("t2s_encoder : ")
        print(t2s_encoder.get_summary())
        print("t2s_first_decoder : ")
        print(t2s_first_decoder.get_summary())
        print("t2s_stage_decoder : ")
        print(t2s_stage_decoder.get_summary())
        if not args.onnx_vits:
            print("vits : ")
            print(vits.get_summary())


if __name__ == '__main__':
    main()
