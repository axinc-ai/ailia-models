import time
import sys
import argparse
import re

import numpy as np
import soundfile as sf

import ailia  # noqa: E402

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from scipy.io.wavfile import write
# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

import torch
import torchaudio
from torch import nn

import onnxruntime
import LangSegment

import os
from text import cleaned_text_to_sequence
from text.japanese import g2p
import soundfile

import ffmpeg
import numpy as np
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
    '--input', '-i', metavar='TEXT', default="ax株式会社ではAIの実用化のためのフレームワークを開発しています。",
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
    '--profile', action='store_true',
    help='use profile model'
)
args = update_parser(parser, check_input_type=False)

WEIGHT_PATH_SSL = 'nahida_cnhubert.onnx'
WEIGHT_PATH_T2S_ENCODER = 'nahida_t2s_encoder.onnx'
WEIGHT_PATH_T2S_FIRST_DECODER = 'nahida_t2s_fsdec.onnx'
WEIGHT_PATH_T2S_STAGE_DECODER = 'nahida_t2s_sdec.onnx'
WEIGHT_PATH_VITS = 'nahida_vits.onnx'

MODEL_PATH_SSL = None#'nahida_cnhubert.onnx'
MODEL_PATH_T2S_ENCODER = None#'nahida_t2s_encoder.onnx'
MODEL_PATH_T2S_FIRST_DECODER = None#'nahida_t2s_fsdec.onnx'
MODEL_PATH_T2S_STAGE_DECODER = None#'nahida_t2s_sdec.onnx'
MODEL_PATH_VITS = None#'nahida_vits.onnx'



def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


class T2SModel(nn.Module):
    def __init__(self, sess_encoder, sess_fsdec, sess_sdec):
        super().__init__()
        self.hz = 50
        self.max_sec = 54
        self.top_k = 5
        self.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.sess_encoder = sess_encoder
        self.sess_fsdec = sess_fsdec
        self.sess_sdec = sess_sdec

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.early_stop_num

        top_k = torch.LongTensor([5])
        top_p = torch.Tensor([1.0])
        temperature = torch.Tensor([1.0])
        repetition_penalty = torch.Tensor([1.35])

        EOS = 1024

        #[1,N] [1,N] [N, 1024] [N, 1024] [1, 768, N]
        x, prompts = self.sess_encoder.run(None, {"ref_seq":ref_seq.detach().numpy(), "text_seq":text_seq.detach().numpy(), "ref_bert":ref_bert.detach().numpy(), "text_bert":text_bert.detach().numpy(), "ssl_content":ssl_content.detach().numpy()})
        x = torch.from_numpy(x)
        prompts = torch.from_numpy(prompts)

        prefix_len = prompts.shape[1]

        #[1,N,512] [1,N]
        y, k, v, y_emb, x_example = self.sess_fsdec.run(None, {"x":x.detach().numpy(), "prompts":prompts.detach().numpy(), "top_k":top_k.detach().numpy(), "top_p":top_p.detach().numpy(), "temperature":temperature.detach().numpy(), "repetition_penalty":repetition_penalty.detach().numpy()})
        y = torch.from_numpy(y)
        k = torch.from_numpy(k)
        v = torch.from_numpy(v)
        y_emb = torch.from_numpy(y_emb)
        x_example = torch.from_numpy(x_example)

        stop = False
        for idx in range(1, 1500):
            #[1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            y, k, v, y_emb, logits, samples = self.sess_sdec.run(None, {"iy":y.detach().numpy(), "ik":k.detach().numpy(), "iv":v.detach().numpy(), "iy_emb":y_emb.detach().numpy(), "ix_example":x_example.detach().numpy(), "top_k":top_k.detach().numpy(), "top_p":top_p.detach().numpy(), "temperature":temperature.detach().numpy(), "repetition_penalty":repetition_penalty.detach().numpy()})
            y = torch.from_numpy(y)
            k = torch.from_numpy(k)
            v = torch.from_numpy(v)
            y_emb = torch.from_numpy(y_emb)
            logits = torch.from_numpy(logits)
            samples = torch.from_numpy(samples)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == EOS or samples[0, 0] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[:, -idx:-1].unsqueeze(0)





class GptSoVits(nn.Module):
    def __init__(self, t2s, sess):
        super().__init__()
        self.t2s = t2s
        self.sess = sess
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        audio1 = self.sess.run(None, {
            "text_seq" : text_seq.detach().cpu().numpy(),
            "pred_semantic" : pred_semantic.detach().cpu().numpy(), 
            "ref_audio" : ref_audio.detach().cpu().numpy()
        })
        return torch.from_numpy(audio1[0])



class SSLModel(nn.Module):
    def __init__(self, sess):
        super().__init__()
        self.sess = sess

    def forward(self, ref_audio_16k):
        last_hidden_state = self.sess.run(None, {
            "ref_audio_16k" : ref_audio_16k.detach().cpu().numpy()
        })
        return torch.from_numpy(last_hidden_state[0])


def generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits):
    gpt = T2SModel(t2s_encoder, t2s_first_decoder, t2s_stage_decoder,)
    gpt_sovits = GptSoVits(gpt, vits)
    ssl = SSLModel(ssl)

    ref_audio = torch.randn((1, 48000 * 5)).float()

    input_audio = args.audio
    ref_phones = g2p(args.transcript)
    ref_audio = torch.tensor([load_audio(input_audio, 48000)]).float()

    ref_seq = torch.LongTensor([cleaned_text_to_sequence(ref_phones)])

    text_phones = g2p(args.input)
    print(text_phones)
    text_seq = torch.LongTensor([cleaned_text_to_sequence(text_phones)])

    # empty for ja or en
    ref_bert = torch.zeros((ref_seq.shape[1], 1024)).float()
    text_bert = torch.zeros((text_seq.shape[1], 1024)).float()

    ref_audio_16k = torchaudio.functional.resample(ref_audio,48000,16000).float()
    vits_hps_data_sampling_rate = 32000
    ref_audio_sr = torchaudio.functional.resample(ref_audio,48000,vits_hps_data_sampling_rate).float()

    zero_wav = np.zeros(
        int(vits_hps_data_sampling_rate * 0.3),
        dtype=np.float32,
    )
    wav16k, sr = librosa.load(input_audio, sr=16000)
    wav16k = torch.from_numpy(wav16k)
    zero_wav_torch = torch.from_numpy(zero_wav)
    wav16k = torch.cat([wav16k, zero_wav_torch]).unsqueeze(0)
    ref_audio_16k = wav16k # hubertの入力のみpaddingする
    
    ssl_content = ssl(ref_audio_16k).float()

    a = gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content)

    savepath = args.savepath
    logger.info(f'saved at : {savepath}')

    soundfile.write(savepath, a.cpu().detach().numpy(), vits_hps_data_sampling_rate)

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
        ssl = onnxruntime.InferenceSession(WEIGHT_PATH_SSL)
        t2s_encoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_ENCODER)
        t2s_first_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_FIRST_DECODER)
        t2s_stage_decoder = onnxruntime.InferenceSession(WEIGHT_PATH_T2S_STAGE_DECODER)
        vits = onnxruntime.InferenceSession(WEIGHT_PATH_VITS)
    else:
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        ssl = ailia.Net(stream = WEIGHT_PATH_T2S_ENCODER, weight = MODEL_PATH_SSL, memory_mode = memory_mode, env_id = args.env_id)
        t2s_encoder = ailia.Net(stream = WEIGHT_PATH_T2S_ENCODER, weight = MODEL_PATH_T2S_ENCODER, memory_mode = memory_mode, env_id = args.env_id)
        t2s_first_decoder = ailia.Net(stream = WEIGHT_PATH_T2S_FIRST_DECODER, weight = MODEL_PATH_T2S_FIRST_DECODER, memory_mode = memory_mode, env_id = args.env_id)
        t2s_stage_decoder = ailia.Net(stream = WEIGHT_PATH_T2S_STAGE_DECODER, weight = MODEL_PATH_T2S_STAGE_DECODER, memory_mode = memory_mode, env_id = args.env_id)
        vits = ailia.Net(stream = WEIGHT_PATH_VITS, weight = MODEL_PATH_VITS, memory_mode = memory_mode, env_id = args.env_id)
        if args.profile:
            ssl.set_profile_mode(True)
            t2s_encoder.set_profile_mode(True)
            t2s_first_decoder.set_profile_mode(True)
            t2s_stage_decoder.set_profile_mode(True)
            vits.set_profile_mode(True)

    generate_voice(ssl, t2s_encoder, t2s_first_decoder, t2s_stage_decoder, vits)

    if args.profile:
        print("ssl : ")
        ssl(ssl.get_summary())
        print("t2s_encoder : ")
        print(t2s_encoder.get_summary())
        print("t2s_first_decoder : ")
        print(t2s_first_decoder.get_summary())
        print("t2s_stage_decoder : ")
        print(t2s_stage_decoder.get_summary())
        print("vits : ")
        print(vits.get_summary())


if __name__ == '__main__':
    main()
