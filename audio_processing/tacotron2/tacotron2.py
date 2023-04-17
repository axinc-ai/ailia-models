import time
import sys
import argparse

import numpy as np
import soundfile as sf

sys.path.append('text/')
from text import text_to_sequence

import ailia  # noqa: E402

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

SAVE_WAV_PATH = 'output.wav'

WEIGHT_PATH_DECODER_ITER = 'decoder_iter.onnx'
MODEL_PATH_DECODER_ITER  = 'decoder_iter.onnx.prototxt'
WEIGHT_PATH_ENCODER = 'encoder.onnx'
MODEL_PATH_ENCODER  = 'encoder.onnx.prototxt'
WEIGHT_PATH_POSTNET = 'postnet.onnx'
MODEL_PATH_POSTNET  = 'postnet.onnx.prototxt'

WEIGHT_PATH_WAVEGLOW = 'waveglow.onnx'
MODEL_PATH_WAVEGLOW  = 'waveglow.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/tacotron2/'


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser( 'Tacotron2', None, SAVE_WAV_PATH)
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
    '--japanese', action='store_true',
    help='use japanese model'
)
args = update_parser(parser, check_input_type=False)

WEIGHT_PATH_DECODER_ITER = 'tsukuyomi_decoder_iter.onnx'
WEIGHT_PATH_ENCODER = 'tsukuyomi_encoder.onnx'
WEIGHT_PATH_POSTNET = 'tsukuyomi_postnet.onnx'

# ======================
# Parameters
# ======================

args_onnx = args.onnx

if args_onnx:
    import onnxruntime
    print("Use onnx runtime.")
else:
    import ailia
    print("Use ailia.")

if args.japanese:
    model_name ="tsukuyomi"
else:
    model_name ="nvidia"

if model_name == "nvidia":
    if args.input:
        text = args.input
    else:
        text = "hello world. we will introduce new AI engine ailia. ailia is high speed inference engine."
else:
    text ="こんにちは。今日は新しいAIエンジンであるアイリアSDKを紹介します。アイリアSDKは高速なAI推論エンジンです。"
    import pyopenjtalk
    text = pyopenjtalk.g2p(text, kana=False)
    text = text.replace('pau',',')
    text = text.replace(' ','')
    text = text + '.'


sampling_rate = 22050

# ======================
# Functions
# ======================

def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths = np.sort([len(x) for x in batch])
    input_lengths = input_lengths[::-1].copy()
    ids_sorted_decreasing = np.argsort([len(x) for x in batch])
    ids_sorted_decreasing = ids_sorted_decreasing[::-1].copy()

    max_input_len = input_lengths[0]

    text_padded = np.zeros((len(batch), max_input_len), dtype=np.int64)
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :len(text)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=False):
    # Convert text to sequence
    d = []
    for i,text in enumerate(texts):
        d.append(
            text_to_sequence(text, ['english_cleaners'])[:])

    # Padding to max length of all batches
    text_padded, input_lengths = pad_sequences(d)
    
    return text_padded, input_lengths

def get_mask_from_lengths(lengths_in):
    # Create enable mask for input sequence to care batch size
    lengths = lengths_in
    max_len = np.max(lengths)
    ids = np.arange(0, max_len, dtype=lengths.dtype)
    mask = (ids < np.expand_dims(lengths, 1))
    mask = np.less_equal(mask, 0)
    return mask

def init_decoder_inputs(memory, processed_memory, memory_lengths):

    dtype = np.float32
    bs = memory.shape[0]
    seq_len = memory.shape[1]
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = np.zeros((bs, attention_rnn_dim)).astype(dtype)
    attention_cell = np.zeros((bs, attention_rnn_dim)).astype(dtype)
    decoder_hidden = np.zeros((bs, decoder_rnn_dim)).astype(dtype)
    decoder_cell = np.zeros((bs, decoder_rnn_dim)).astype(dtype)
    attention_weights = np.zeros((bs, seq_len)).astype(dtype)
    attention_weights_cum = np.zeros((bs, seq_len)).astype(dtype)
    attention_context = np.zeros((bs, encoder_embedding_dim)).astype(dtype)
    mask = get_mask_from_lengths(memory_lengths)
    decoder_input = np.zeros((bs, n_mel_channels)).astype(dtype)

    return (decoder_input, attention_hidden, attention_cell, decoder_hidden,
            decoder_cell, attention_weights, attention_weights_cum,
            attention_context, memory, processed_memory, mask)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_inference(texts, encoder, decoder_iter, postnet):
    #print("Running Tacotron2 Encoder")
    sequences, sequence_lengths = prepare_input_sequence(texts)
    if args_onnx:
        encoder_inputs = {encoder.get_inputs()[0].name: sequences,
                        encoder.get_inputs()[1].name: sequence_lengths}
        encoder_outs = encoder.run(None, encoder_inputs)
    else:
        encoder_inputs = [sequences,
                          sequence_lengths]
        encoder_outs = encoder.run(encoder_inputs)
    memory, processed_memory, lens = encoder_outs

    #print("Running Tacotron2 Decoder")
    mel_lengths = np.zeros([memory.shape[0]], dtype=np.int32)
    not_finished = np.ones([memory.shape[0]], dtype=np.int32)
    mel_outputs, gate_outputs, alignments = (np.zeros(1), np.zeros(1), np.zeros(1))
    gate_threshold = 0.6
    max_decoder_steps = 1000
    first_iter = True

    (decoder_input, attention_hidden, attention_cell, decoder_hidden,
     decoder_cell, attention_weights, attention_weights_cum,
     attention_context, memory, processed_memory,
     mask) = init_decoder_inputs(memory, processed_memory, sequence_lengths)

    while True:
        if args_onnx:
            decoder_inputs = {decoder_iter.get_inputs()[0].name: decoder_input,
                            decoder_iter.get_inputs()[1].name: attention_hidden,
                            decoder_iter.get_inputs()[2].name: attention_cell,
                            decoder_iter.get_inputs()[3].name: decoder_hidden,
                            decoder_iter.get_inputs()[4].name: decoder_cell,
                            decoder_iter.get_inputs()[5].name: attention_weights,
                            decoder_iter.get_inputs()[6].name: attention_weights_cum,
                            decoder_iter.get_inputs()[7].name: attention_context,
                            decoder_iter.get_inputs()[8].name: memory,
                            decoder_iter.get_inputs()[9].name: processed_memory,
                            decoder_iter.get_inputs()[10].name: mask
                            }
            decoder_outs = decoder_iter.run(None, decoder_inputs)
        else:
            decoder_inputs = [decoder_input,
                            attention_hidden,
                            attention_cell,
                            decoder_hidden,
                            decoder_cell,
                            attention_weights,
                            attention_weights_cum,
                            attention_context,
                            memory,
                            processed_memory,
                            mask
                            ]
            decoder_outs = decoder_iter.run(decoder_inputs)
        (mel_output, gate_output,
        attention_hidden, attention_cell,
        decoder_hidden, decoder_cell,
        attention_weights, attention_weights_cum,
        attention_context) = decoder_outs

        # Generated one mel_output (80, 1) from one decode

        if first_iter:
            mel_outputs = np.expand_dims(mel_output, 2)
            gate_outputs = np.expand_dims(gate_output, 2)
            alignments = np.expand_dims(attention_weights, 2)
            first_iter = False
        else:
            mel_outputs = np.concatenate([mel_outputs, np.expand_dims(mel_output, 2)], axis = 2)
            gate_outputs = np.concatenate([gate_outputs, np.expand_dims(gate_output, 2)], axis = 2)
            alignments = np.concatenate([alignments, np.expand_dims(attention_weights, 2)], axis = 2)

        dec = np.less_equal(sigmoid(gate_output), gate_threshold).astype(np.int32).squeeze(1)
        not_finished = not_finished*dec
        mel_lengths += not_finished

        if np.sum(not_finished) == 0:
            print("Stopping after ",mel_outputs.shape[2]," decoder steps")
            break
        if mel_outputs.shape[2] == max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break

        decoder_input = mel_output

    #print("Running Tacotron2 PostNet")
    if args_onnx:
        postnet_inputs = {postnet.get_inputs()[0].name: mel_outputs}
        mel_outputs_postnet = postnet.run(None, postnet_inputs)[0]
    else:
        postnet_inputs = [mel_outputs]
        mel_outputs_postnet = postnet.run(postnet_inputs)[0]

    return mel_outputs_postnet

def generate_voice(decoder_iter, encoder, postnet, waveglow):
    # onnx
    logger.info("Input text : ", text)

    texts = [text]

    mel_outputs_postnet = test_inference(texts, encoder, decoder_iter, postnet)

    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel_outputs_postnet.shape[2]*stride)//n_group
    z = np.random.randn(1, n_group, z_size2).astype(np.float32)

    #print("Running Tacotron2 Waveglow")

    if args_onnx:
        waveglow_inputs = {waveglow.get_inputs()[0].name: mel_outputs_postnet,
                        waveglow.get_inputs()[1].name: z}
        audio = waveglow.run(None, waveglow_inputs)[0]
    else:
        waveglow_inputs = [mel_outputs_postnet, z]
        audio = waveglow.run(waveglow_inputs)[0]

    # export to audio
    savepath = args.savepath
    logger.info(f'saved at : {savepath}')
    sf.write(savepath, audio[0].astype(np.float32), sampling_rate)
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_DECODER_ITER, MODEL_PATH_DECODER_ITER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_ENCODER, MODEL_PATH_ENCODER, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_POSTNET, MODEL_PATH_POSTNET, REMOTE_PATH)
    check_and_download_models(WEIGHT_PATH_WAVEGLOW, MODEL_PATH_WAVEGLOW, REMOTE_PATH)

    #env_id = args.env_id

    if args_onnx:
        decoder_iter = onnxruntime.InferenceSession(WEIGHT_PATH_DECODER_ITER)
        encoder = onnxruntime.InferenceSession(WEIGHT_PATH_ENCODER)
        postnet = onnxruntime.InferenceSession(WEIGHT_PATH_POSTNET)
        waveglow = onnxruntime.InferenceSession(WEIGHT_PATH_WAVEGLOW)
    else:
        memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True, reduce_interstage=False, reuse_interstage=True)
        decoder_iter = ailia.Net(stream = MODEL_PATH_DECODER_ITER, weight = WEIGHT_PATH_DECODER_ITER, memory_mode = memory_mode, env_id = args.env_id)
        encoder = ailia.Net(stream = MODEL_PATH_ENCODER, weight = WEIGHT_PATH_ENCODER, memory_mode = memory_mode, env_id = args.env_id)
        postnet = ailia.Net(stream = MODEL_PATH_POSTNET, weight = WEIGHT_PATH_POSTNET, memory_mode = memory_mode, env_id = args.env_id)
        waveglow = ailia.Net(stream = MODEL_PATH_WAVEGLOW, weight = WEIGHT_PATH_WAVEGLOW, memory_mode = memory_mode, env_id = args.env_id)

    generate_voice(decoder_iter, encoder, postnet, waveglow)


if __name__ == '__main__':
    main()
