import os
import sys
import json
import time
import queue
import threading

import numpy as np

# Qwen3-TTS streaming on Core ML (Apple Neural Engine), macOS 15 or later.
#
# The same live text loop as qwen3-tts-streaming.py (Base voice clone, 0.6B),
# with the models in Core ML: the export of export/export_coreml.py (--coreml_dir),
# or the build https://huggingface.co/tinytrashlabs/Qwen3-TTS-0.6B-Base-ANE it
# follows the layout of (downloaded when --coreml_dir does not exist):
#   prompt            text token ids -> talker hidden: 4 bit text embedding + projection MLP, numpy
#   codec_embedding   the 16 codec tables, numpy
#   talker            talker0 / talker1.mlmodelc, 14 layers each, KV cache in a Core ML state, ANE
#   code_predictor    cp_ane.mlmodelc, groups 1..15 of a frame in one call, ANE (the export samples
#                     inside the graph with noise from here, the tinytrashlabs build picks the argmax)
#   decoder           vocoder head (RVQ + pre_conv + 8 layer transformer), numpy, then
#                     upF.mlmodelc (CPU) and upMall.mlmodelc (GPU, see --vocoder_units)
#   encoder           the ONNX reference encoder of qwen3-tts-streaming (ailia or onnxruntime):
#                     the Core ML speech encoder of the tinytrashlabs build degenerates after ~2.7 s on M2
# The model classes keep the interface of qwen3-tts-streaming.py (prompt.run,
# talker.prefill / step, ...).

# import original modules (ailia-models layout); fall back to small shims so the
# sample also runs outside ailia-models
sys.path.append('../../util')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from arg_utils import get_base_parser, update_parser  # noqa: E402
    from model_utils import check_and_download_models, check_and_download_file  # noqa: E402
except ImportError:  # not inside ailia-models
    from ailia_shim import get_base_parser, update_parser, check_and_download_models, check_and_download_file  # noqa: E402
try:
    import ailia
    AILIA_EXIST = True
except ImportError:
    AILIA_EXIST = False

from live_text_source import LiveTextSource, token_offsets_fn  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================

OUTPUT_WAV_PATH = 'output.wav'
INPUT_TEXT_PATH = 'input.txt'
REFERENCE_WAV_PATH = 'reference.wav'
REFERENCE_TEXT = "かしこまりました。"
UTTERANCE_END = "。！？!?"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-tts-streaming/"
TOKENIZER_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/qwen3-tts/"   # same tokenizer as qwen3-tts
COREML_REPO = "tinytrashlabs/Qwen3-TTS-0.6B-Base-ANE"
COREML_FILES = ["coreml/talker0.mlmodelc/*", "coreml/talker1.mlmodelc/*", "coreml/cp_ane.mlmodelc/*",
                "coreml/upF.mlmodelc/*", "coreml/upMall.mlmodelc/*", "host/*", "vochead/*"]

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920        # decoder upsample rate: 24000 Hz / 12.5 frames per second
SILENCE_RMS = 0.003         # an 80 ms frame quieter than this is silence (speech is 0.01 - 0.06)
TAIL_SILENCE_S = 3.0        # silence after the whole text that ends an utterance: the text is all in
                            # well before it is all spoken, and pauses of 2 s occur inside a sentence
KEEP_TAIL_S = 0.2           # of which this much is kept

CONFIG_PATH = "config_0.6B.json"
WEIGHT_PATH_ENCODER = "qwen3_tts_encoder_0.6B.onnx"
MODEL_PATH_ENCODER = WEIGHT_PATH_ENCODER + ".prototxt"

TOKENIZER_DIR = "./tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "vocab.json")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "merges.txt")
TOKENIZER_SPECIAL_TOKENS = [
    '<|endoftext|>', '<|im_start|>', '<|im_end|>',
    '<|object_ref_start|>', '<|object_ref_end|>',
    '<|box_start|>', '<|box_end|>',
    '<|quad_start|>', '<|quad_end|>',
    '<|vision_start|>', '<|vision_end|>',
    '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>',
]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Qwen3-TTS streaming on Core ML (Base voice clone, 0.6B): text in, audio out, both incremental',
    INPUT_TEXT_PATH,
    OUTPUT_WAV_PATH,
    large_model=True,
)
parser.add_argument('--ref_audio', default=REFERENCE_WAV_PATH, help='reference audio to clone the voice of')
parser.add_argument('--ref_text', default=REFERENCE_TEXT,
                    help='transcript of the reference audio (in-context voice clone). An empty string '
                         'selects the speaker-embedding-only clone (x_vector_only_mode of Qwen3-TTS)')
parser.add_argument(
    '--language', default='Japanese',
    help='Auto, or chinese, english, japanese, korean, german, french, russian, portuguese, spanish, italian'
)
parser.add_argument(
    '--commit', default='token', choices=['token', 'punct'],
    help='when text becomes available to the talker: token = per talker token, 3 held back (default, earliest '
         'speech); punct = at spaces and 、。！？ etc. (tokenizes exactly like the whole text)'
)
parser.add_argument(
    '--cps', type=float, default=30.0,
    help='the input text is pushed in pieces of --piece characters at this many characters per second, like an LLM would'
)
parser.add_argument('--piece', type=int, default=2, help='characters per push')
parser.add_argument(
    '--max_chars', type=int, default=150,
    help='the text is spoken as consecutive utterances cut at 。！？ or line ends, each at most this long '
         '(the talker KV cache holds 1024 positions, about 75 s with the prompt)'
)
# how the audio leaves
parser.add_argument('--play', action='store_true', help='play on the default audio device (needs sounddevice)')
parser.add_argument('--quiet', action='store_true', help='do not echo the text to the console as it is fed in')
parser.add_argument('--prebuffer', type=float, default=0.5, help='seconds of audio to queue before playback starts')
parser.add_argument('--results', default=None, help='append one JSON line of timings per utterance to this file')
# generation
parser.add_argument('--temperature', type=float, default=0.9, help='talker sampling temperature, 0 = greedy')
parser.add_argument('--top_k', type=int, default=50, help='talker top-k')
parser.add_argument('--subtalker_temperature', type=float, default=0.9,
                    help='code predictor temperature, 0 = greedy (a code predictor with a noise input only)')
parser.add_argument(
    '--emit_every', type=int, default=12,
    help='frames per audio chunk, 1..12 (the upsampler decodes windows of 12 new frames; a shorter chunk is '
         'zero padded and costs a whole window)'
)
parser.add_argument('--first_chunk_emit', type=int, default=0, help='frames in the first chunk, 0 = same as --emit_every')
parser.add_argument('--seed', type=int, default=None, help='random seed')
# runtime
parser.add_argument('--coreml_dir', default='coreml_ane', help='where the Core ML build is downloaded to')
parser.add_argument(
    '--vocoder_units', default='gpu', choices=['gpu', 'ane', 'cpu'],
    help='compute units of the upsampler (upMall). ane is what the model was built for, but on M2 / macOS 15 '
         'its output is noise (SNR -3.4 dB against the reference); gpu is correct and fast'
)
parser.add_argument('--onnx', action='store_true', help='run the reference encoder on onnxruntime instead of ailia')
parser.add_argument('--disable_ailia_tokenizer', action='store_true', help='disable ailia tokenizer.')
if __name__ == '__main__':
    # only the main process parses the arguments: the decoder process (spawned)
    # imports this file as __mp_main__ and needs none of them
    args = update_parser(parser, check_input_type=False)

    if not AILIA_EXIST and not args.onnx:
        logger.info("ailia is not installed, running the encoder on onnxruntime")
        args.onnx = True


# ======================
# Helpers
# ======================
class Benchmark:
    """Per model call counts and total time, reported once at the end (-b). Thread safe:
    the decoder runs on its own thread."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = {}
        self.lock = threading.Lock()

    def add(self, name, seconds):
        with self.lock:
            calls, total = self.records.get(name, (0, 0.0))
            self.records[name] = (calls + 1, total + seconds)

    def reset(self):
        with self.lock:
            self.records = {}

    def report(self):
        for name, (calls, total) in self.records.items():
            logger.info("\t{} processing time {:.0f} ms ({} calls, {:.1f} ms/call)".format(
                name, total * 1000, calls, total * 1000 / calls))

    def as_dict(self):
        return {k: [c, round(t * 1000 / c, 2)] for k, (c, t) in self.records.items()}


def top_k_top_p(logits, top_k=0, top_p=1.0):
    if 0 < top_k < logits.shape[-1]:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        probs = np.exp(sorted_logits - sorted_logits.max())
        probs /= probs.sum()
        remove = np.cumsum(probs) > top_p
        remove[0] = False
        sorted_logits[remove] = -np.inf
        logits = np.empty_like(sorted_logits)
        logits[order] = sorted_logits
    return logits


def sample_token(logits, rng, temperature=0.9, top_k=50, top_p=1.0, suppress=None):
    """One token id from a 1D logits vector; temperature 0 is greedy."""
    logits = logits.astype(np.float64, copy=True)
    if suppress is not None:
        logits[suppress] = -np.inf
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = top_k_top_p(logits / temperature, top_k, top_p)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# mel spectrogram fed to the speaker encoder (as in audio_processing/qwen3-tts)
MEL_N_FFT, MEL_NUM_MELS, MEL_HOP, MEL_WIN, MEL_FMIN, MEL_FMAX = 1024, 128, 256, 1024, 0, 12000


def mel_spectrogram(y, sr):
    import librosa
    mel_basis = librosa.filters.mel(sr=sr, n_fft=MEL_N_FFT, n_mels=MEL_NUM_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)
    padding = (MEL_N_FFT - MEL_HOP) // 2
    y = np.pad(y[None], [(0, 0), (padding, padding)], mode="reflect")
    spec = np.abs(np.squeeze(librosa.stft(y, n_fft=MEL_N_FFT, hop_length=MEL_HOP, win_length=MEL_WIN,
                                          window=np.hanning(MEL_WIN), center=False, pad_mode="reflect")))
    spec = np.log(np.clip(np.dot(mel_basis, spec), 1e-5, None)).T          # (T, 128)
    return spec[None].astype(np.float32)


def create_tokenizer():
    """(encode function, tokenizer): the tokenizer also gives LiveTextSource its token offsets."""
    if args.disable_ailia_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        return lambda text: tokenizer(text)["input_ids"], tokenizer
    from ailia_tokenizer import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.add_special_tokens({"additional_special_tokens": TOKENIZER_SPECIAL_TOKENS})
    return lambda text: list(tokenizer.encode(text)), tokenizer


def silu(x):
    return x / (1.0 + np.exp(-x))


def rms_norm(x, w, eps=1e-5):
    return w * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))


# ======================
# Reference encoder (ONNX)
# ======================
class EncoderNet:
    """The fp32 ONNX reference encoder of qwen3-tts-streaming on ailia or onnxruntime."""

    def __init__(self, env_id, bench):
        self.bench = bench
        if args.onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(WEIGHT_PATH_ENCODER, providers=["CPUExecutionProvider"])
            self.net = None
        else:
            memory_mode = ailia.get_memory_mode(reduce_constant=True, ignore_input_with_initializer=True,
                                                reduce_interstage=False, reuse_interstage=True)
            self.net = ailia.Net(stream=MODEL_PATH_ENCODER, weight=WEIGHT_PATH_ENCODER,
                                 memory_mode=memory_mode, env_id=env_id)

    def run(self, *inputs):
        t = time.perf_counter()
        if self.net is None:
            names = [i.name for i in self.session.get_inputs()]
            out = self.session.run(None, dict(zip(names, inputs)))
        else:
            for index, value in enumerate(inputs):
                self.net.set_input_blob_shape(value.shape, index)
            out = self.net.run(list(inputs))
        self.bench.add("encoder", time.perf_counter() - t)
        return out


# ======================
# Core ML models
# ======================
class PromptNet:
    """text token ids -> talker hidden, as qwen3_tts_prompt: the text embedding
    (4 bit, groups of 64, scale and bias per group) and the 2 layer projection MLP."""

    def __init__(self, host_dir, bench):
        ld = lambda n: np.load(os.path.join(host_dir, n + ".npy"), mmap_mode="r")
        self.q = ld("text_embedding_q")                         # [vocab, 256] uint32, 8 codes a word
        self.scales = ld("text_embedding_scales")               # [vocab, 32] fp16
        self.biases = ld("text_embedding_biases")
        self.w1 = np.asarray(ld("text_proj_linear_fc1_w"), np.float32)
        self.b1 = np.asarray(ld("text_proj_linear_fc1_b"), np.float32)
        self.w2 = np.asarray(ld("text_proj_linear_fc2_w"), np.float32)
        self.b2 = np.asarray(ld("text_proj_linear_fc2_b"), np.float32)
        self.shifts = (np.arange(8, dtype=np.uint32) * 4)
        self.bench = bench

    def run(self, ids):
        """ids [1, n] -> [hidden [1, n, H]]"""
        t = time.perf_counter()
        ids = np.asarray(ids).reshape(-1)
        codes = (self.q[ids][:, :, None] >> self.shifts) & 0xF                 # [n, 256, 8]
        codes = codes.reshape(len(ids), -1, 64).astype(np.float32)            # [n, 32, 64]
        x = codes * self.scales[ids][:, :, None].astype(np.float32) + self.biases[ids][:, :, None].astype(np.float32)
        x = x.reshape(len(ids), -1)                                            # [n, 2048]
        y = silu(x @ self.w1.T + self.b1) @ self.w2.T + self.b2
        self.bench.add("prompt", time.perf_counter() - t)
        return [y[None].astype(np.float32)]


class CodecEmbedding:
    """The 16 codec tables in the row layout of qwen3_tts_codec_embedding: the
    talker's 3072 rows, then 15 x 2048 code predictor rows, then one zero row."""

    def __init__(self, host_dir):
        talker = np.load(os.path.join(host_dir, "talker_codec_embedding.npy")).astype(np.float32)
        cp = np.load(os.path.join(host_dir, "cp_codec_embedding.npy")).astype(np.float32)
        self.table = np.concatenate([talker, cp.reshape(-1, cp.shape[-1]),
                                     np.zeros((1, talker.shape[1]), np.float32)])


class CoreMLTalker:
    """The 28 layer talker as two stateful Core ML chunks (layers 0-13 / 14-27),
    each with a "decode" (1 row) and a "prefill" (64 rows) function sharing one
    MLState: the KV cache, 1024 positions of fp16 (8, 1024, 128) per layer.

    The ANE has no scatter, so a row is written by a one hot blend: `place` puts
    the new key / value at a position and `keep` clears it in the old cache.
    Chunk 0 hands its residual to chunk 1 as (x_out, lo_out), a compensated fp16
    pair. The KV cache is kept across calls, so a prefix prefilled once stays
    valid (prefix_key, as in qwen3-tts-streaming.py); rows past the current
    position are masked and rewritten before they are used."""

    P, HD = 64, 128

    def __init__(self, coreml_dir, bench, hidden):
        import coremltools as ct
        units = ct.ComputeUnit.CPU_AND_NE
        path = lambda i: os.path.join(coreml_dir, f"talker{i}.mlmodelc")
        self.dec = [ct.models.CompiledMLModel(path(i), units, function_name="decode") for i in range(2)]
        self.pre = [ct.models.CompiledMLModel(path(i), units, function_name="prefill") for i in range(2)]
        self.states = [m.make_state() for m in self.dec]
        self.buf = 1024
        self.hidden = hidden
        self.bench = bench
        self.prefix_key = None
        # RoPE rows for every position, theta 1e6, rounded once to fp16
        inv = 1.0 / np.power(1e6, np.arange(0, self.HD, 2, dtype=np.float64) / self.HD)
        ang = np.arange(self.buf, dtype=np.float64)[:, None] * inv[None]
        self.cos = np.tile(np.cos(ang), 2).astype(np.float16)
        self.sin = np.tile(np.sin(ang), 2).astype(np.float16)
        self.cols = np.arange(self.buf)

    def _run(self, models, feed):
        out = None
        for model, state in zip(models, self.states):
            out = model.predict(feed, state=state)
            if "x_out" in out:
                feed = dict(feed, x=out["x_out"].astype(np.float16), lo=out["lo_out"].astype(np.float16))
        return out

    def _prefill_rows(self, embeds, start):
        """Writes positions start.. with the prefill function, 64 rows a call (no output)."""
        P, L = self.P, self.buf
        n = embeds.shape[0]
        for s in range(0, n, P):
            rows = min(P, n - s)
            pos = start + s + np.arange(P)
            valid = pos < L
            x = np.zeros((P, self.hidden), np.float16)
            x[:rows] = embeds[s:s + rows]
            mask = np.where(self.cols[None] <= pos[:, None], 0, -10000).astype(np.float16)
            place = np.zeros((L, P), np.float16)
            place[pos[valid], np.arange(P)[valid]] = 1
            keep = np.ones((L, 1), np.float16)
            keep[pos[valid]] = 0
            p = np.minimum(pos, L - 1)
            self._run(self.pre, dict(x=x, cos=self.cos[p], sin=self.sin[p], mask=mask, place=place, keep=keep))

    def prefill(self, inputs_embeds, start=0):
        """Feed positions start..; returns (logits, last_hidden) of the last one."""
        t = time.perf_counter()
        e = inputs_embeds[0].astype(np.float16)
        if e.shape[0] > 1:
            self._prefill_rows(e[:-1], start)
        self.bench.add("talker_prefill", time.perf_counter() - t)
        return self.step(inputs_embeds[:, -1:], start + e.shape[0] - 1)

    def step(self, x, position):
        t = time.perf_counter()
        L = self.buf
        mask = np.where(self.cols <= position, 0, -10000).astype(np.float16)[None]
        place = np.zeros((L, 1), np.float16)
        place[position] = 1
        keep = np.ones((L, 1), np.float16)
        keep[position] = 0
        out = self._run(self.dec, dict(
            x=np.asarray(x, np.float16).reshape(1, self.hidden), cos=self.cos[position:position + 1],
            sin=self.sin[position:position + 1], mask=mask, place=place, keep=keep))
        logits = out["logits"].astype(np.float32).reshape(1, 1, -1)
        if not np.isfinite(logits).all():
            raise RuntimeError(f"talker logits at position {position} are not finite")
        self.bench.add("talker", time.perf_counter() - t)
        return logits, out["hidden"].astype(np.float32).reshape(1, 1, -1)


class CoreMLCodePredictor:
    """Groups 1..15 of a frame from the talker's hidden state and the group 0
    embedding, one call, the 15 steps unrolled in the graph.

    The export of export/coreml samples inside the graph:
    argmax(top_50(logits) + noise), with noise = temperature * Gumbel noise from
    here (--subtalker_temperature, 0 = greedy). A model without the noise input
    (the tinytrashlabs build) always picks the argmax."""

    def __init__(self, coreml_dir, bench, rng, temperature):
        import coremltools as ct
        path = os.path.join(coreml_dir, "cp_ane.mlmodelc")
        self.model = ct.models.CompiledMLModel(path, ct.ComputeUnit.CPU_AND_NE)
        meta = json.load(open(os.path.join(path, "metadata.json")))[0]
        self.samples = any(i["name"] == "noise" for i in meta.get("inputSchema", []))
        self.bench, self.rng, self.temperature = bench, rng, temperature

    def run(self, past_hidden, e0):
        t = time.perf_counter()
        feed = dict(hidden=np.asarray(past_hidden, np.float16).reshape(1, -1),
                    e0=np.asarray(e0, np.float16).reshape(1, -1))
        if self.samples:
            if self.temperature > 0:
                u = self.rng.uniform(1e-10, 1.0 - 1e-7, (15, 2048))
                feed["noise"] = (self.temperature * -np.log(-np.log(u))).astype(np.float16)
            else:
                feed["noise"] = np.zeros((15, 2048), np.float16)
        codes = self.model.predict(feed)["codes"]
        self.bench.add("code_predictor", time.perf_counter() - t)
        if not np.isfinite(codes).all():
            raise RuntimeError("code predictor output is not finite")
        return [int(c) for c in np.clip(np.rint(codes.reshape(-1)), 0, 2047)]


class VocoderHead:
    """Codec frames -> upsampler input, streaming and exact: RVQ dequantize,
    causal pre_conv (keeps its last 2 input frames) and the 8 layer causal
    transformer (keeps every past key and value)."""

    LAYERS, HEADS, HEAD_DIM = 8, 16, 64
    WINDOW = 72         # the transformer's sliding window (speech tokenizer config sliding_window)

    def __init__(self, vochead_dir):
        ld = lambda n: np.load(os.path.join(vochead_dir, n + ".npy")).astype(np.float32)
        self.embed = [ld("quantizer.rvq_first.vq.layers.0._codebook.embed")] + \
                     [ld(f"quantizer.rvq_rest.vq.layers.{i}._codebook.embed") for i in range(15)]
        # contiguous [in, out] copies: a strided slice would keep numpy's matmul off BLAS
        self.w_first = np.ascontiguousarray(ld("quantizer.rvq_first.output_proj.weight")[:, :, 0].T)    # [256, 512]
        self.w_rest = np.ascontiguousarray(ld("quantizer.rvq_rest.output_proj.weight")[:, :, 0].T)
        pre_w = ld("pre_conv.conv.weight")                                                             # [1024, 512, 3]
        self.pre_w = [np.ascontiguousarray(pre_w[:, :, k].T) for k in range(3)]                       # 3 x [512, 1024]
        self.pre_b = ld("pre_conv.conv.bias")
        self.in_w, self.in_b = ld("pre_transformer.input_proj.weight"), ld("pre_transformer.input_proj.bias")
        self.norm_w = ld("pre_transformer.norm.weight")
        self.out_w, self.out_b = ld("pre_transformer.output_proj.weight"), ld("pre_transformer.output_proj.bias")
        self.layers = []
        for i in range(self.LAYERS):
            p = f"pre_transformer.layers.{i}."
            self.layers.append({k: ld(p + n) for k, n in [
                ("in_ln", "input_layernorm.weight"), ("post_ln", "post_attention_layernorm.weight"),
                ("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight"),
                ("sa", "self_attn_layer_scale.scale"), ("sm", "mlp_layer_scale.scale")]})
        self.inv_freq = 1.0 / np.power(10000.0, np.arange(0, self.HEAD_DIM, 2, dtype=np.float32) / self.HEAD_DIM)
        self.reset()

    def reset(self):
        self.state = {"conv": np.zeros((2, 512), np.float32), "k": [None] * self.LAYERS,
                      "v": [None] * self.LAYERS, "off": 0}

    def process(self, codes):
        """codes [T, 16] -> hidden [T, 1024]"""
        st = self.state
        codes = np.asarray(codes)
        T = codes.shape[0]
        e1 = self.embed[0][codes[:, 0]]
        er = sum(self.embed[i + 1][codes[:, i + 1]] for i in range(15))
        q = e1 @ self.w_first + er @ self.w_rest                                   # [T, 512]
        qq = np.concatenate([st["conv"], q])                                       # [T + 2, 512]
        st["conv"] = qq[-2:].copy()
        y = sum(qq[k:k + T] @ self.pre_w[k] for k in range(3)) + self.pre_b        # [T, 1024]
        h = y @ self.in_w.T + self.in_b                                            # [T, 512]

        pos = st["off"] + np.arange(T, dtype=np.float32)
        ang = pos[:, None] * self.inv_freq[None]
        cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]                     # [T, 1, 32]
        nh, hd = self.HEADS, self.HEAD_DIM

        def rope(x):
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

        for i, w in enumerate(self.layers):
            a = rms_norm(h, w["in_ln"])
            qh = rope((a @ w["q"].T).reshape(T, nh, hd))
            kh = rope((a @ w["k"].T).reshape(T, nh, hd))
            vh = (a @ w["v"].T).reshape(T, nh, hd)
            K = kh if st["k"][i] is None else np.concatenate([st["k"][i], kh])
            V = vh if st["v"][i] is None else np.concatenate([st["v"][i], vh])     # [S, nh, hd]
            # causal, within the sliding window: query p reads keys p - WINDOW < j <= p
            q_pos = st["off"] + np.arange(T)
            k_pos = st["off"] + T - K.shape[0] + np.arange(K.shape[0])
            hidden = (k_pos[None] > q_pos[:, None]) | (k_pos[None] <= q_pos[:, None] - self.WINDOW)
            scores = np.einsum("thd,shd->hts", qh, K) / np.sqrt(hd)
            scores = np.where(hidden[None], -np.inf, scores)
            st["k"][i], st["v"][i] = K[-(self.WINDOW - 1):], V[-(self.WINDOW - 1):]
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            scores /= scores.sum(-1, keepdims=True)
            att = np.einsum("hts,shd->thd", scores, V).reshape(T, nh * hd)
            h = h + w["sa"] * (att @ w["o"].T)
            a = rms_norm(h, w["post_ln"])
            h = h + w["sm"] * ((silu(a @ w["gate"].T) * (a @ w["up"].T)) @ w["down"].T)
        st["off"] += T
        return rms_norm(h, self.norm_w) @ self.out_w.T + self.out_b


class DecoderCore:
    """Streaming vocoder: VocoderHead (numpy), then upF (Core ML, CPU) and
    upMall (Core ML) on a window of 12 new frames with 8 frames of left context
    (the upsampler's receptive field), so a chunk is exact the moment it exists.
    The first window of a line has no context (W = 12) unless the head was
    primed with the reference frames; the others are W = 20. A short chunk is
    zero padded on the right (causal: its samples do not change)."""

    C, L = 12, 8

    def __init__(self, coreml_dir, vochead_dir, upsampler_units):
        import coremltools as ct
        self.head = VocoderHead(vochead_dir)
        cpu = ct.ComputeUnit.CPU_ONLY
        units = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE, "cpu": cpu}[upsampler_units]
        up_f, up_m = os.path.join(coreml_dir, "upF.mlmodelc"), os.path.join(coreml_dir, "upMall.mlmodelc")
        self.up_f = {w: ct.models.CompiledMLModel(up_f, cpu, function_name=f"w{w}") for w in (12, 20)}
        if {"w12", "w20"} <= self.functions(up_m):
            self.up_m = {w: ct.models.CompiledMLModel(up_m, units, function_name=f"w{w}") for w in (12, 20)}
        else:
            model = ct.models.CompiledMLModel(up_m, units)       # one model, enumerated input widths
            self.up_m = {12: model, 20: model}
        self.primed = {}
        self.begin([], None)

    @staticmethod
    def functions(path):
        meta = json.load(open(os.path.join(path, "metadata.json")))[0]
        return {f["name"] for f in meta.get("functions", [])}

    @staticmethod
    def _copy_state(s):
        return {"conv": s["conv"].copy(), "k": list(s["k"]), "v": list(s["v"]), "off": s["off"]}

    def begin(self, context, key):
        """Starts a line: `context` (the reference frames, [T][16]) primes the head and the window history."""
        if len(context) >= self.L:
            if key not in self.primed:
                # the reference frames go through the head once (their audio is never made):
                # without that the first word starts cold and cracks
                self.head.reset()
                h = self.head.process(np.array(context, np.int64))
                self.primed[key] = (self._copy_state(self.head.state), h[-self.L:].T.copy())
            state, hist = self.primed[key]
            self.head.state = self._copy_state(state)
            self.hist, self.first = hist.copy(), False
        else:
            self.head.reset()
            self.hist, self.first = np.zeros((1024, self.L), np.float32), True

    def chunk(self, codes):
        """codes [n, 16], n <= 12 -> (n * 1920 samples, head seconds, upsampler seconds)"""
        t = time.perf_counter()
        n, C, L = len(codes), self.C, self.L
        h = self.head.process(codes)                                  # [n, 1024]
        head_s = time.perf_counter() - t
        ctx = 0 if self.first else L
        win = np.zeros((1, 1024, ctx + C), np.float32)
        if ctx:
            win[0, :, :ctx] = self.hist
        win[0, :, ctx:ctx + n] = h.T
        if ctx + n >= L:
            self.hist = win[0, :, ctx + n - L:ctx + n].copy()     # the last 8 real frames
        t = time.perf_counter()
        w = ctx + C
        x = self.up_f[w].predict({"hidden": win})["x"]
        wav = self.up_m[w].predict({"x": x.astype(np.float32)})["wav"].reshape(-1)
        up_s = time.perf_counter() - t
        self.first = False
        return wav[ctx * FRAME_SAMPLES:(ctx + n) * FRAME_SAMPLES].astype(np.float32), head_s, up_s


def decoder_process(jobs, results, coreml_dir, vochead_dir, upsampler_units):
    """The decoder's own process. coremltools holds the GIL through most of a
    predict call, so on a thread of the talker's process the head's ~200 small
    numpy operations each wait for a talker / code predictor call to end: a
    45 ms chunk took over a second. A process of its own runs it beside them."""
    core = DecoderCore(coreml_dir, vochead_dir, upsampler_units)
    results.put(("ready", None))
    while True:
        kind, payload = jobs.get()
        try:
            if kind == "begin":
                core.begin(*payload)
            elif kind == "chunk":
                results.put(("wav", core.chunk(payload)))
            elif kind in ("end", "sync"):
                results.put((kind, None))
            elif kind == "quit":
                return
        except Exception as e:  # noqa: BLE001 -- reported by the caller
            results.put(("error", repr(e)))
            if kind in ("end", "sync"):
                results.put((kind, None))


class CoreMLDecoder:
    """The caller side of the decoder process: frames go in as they are made, a
    chunk is sent off every emit_every frames, and ready() / finish() hand the
    audio back in order."""

    C, L = DecoderCore.C, DecoderCore.L

    def __init__(self, coreml_dir, vochead_dir, bench, upsampler_units):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")           # Core ML and fork do not mix
        self.jobs, self.results = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=decoder_process, daemon=True,
                                args=(self.jobs, self.results, coreml_dir, vochead_dir, upsampler_units))
        self.proc.start()
        self.bench = bench
        self.pending = []
        self._wait("ready")

    def _handle(self, kind, payload, chunks):
        if kind == "wav":
            wav, head_s, up_s = payload
            self.bench.add("decoder_head", head_s)
            self.bench.add("decoder_upsampler", up_s)
            chunks.append(wav)
        elif kind == "error":
            logger.error(f"decoder: {payload}")

    def _wait(self, until):
        chunks = []
        while True:
            if not self.proc.is_alive() and self.results.empty():
                raise RuntimeError("the decoder process exited")
            try:
                kind, payload = self.results.get(timeout=1.0)
            except queue.Empty:
                continue
            if kind == until:
                return chunks
            self._handle(kind, payload, chunks)

    def begin(self, context, key, emit_every, first_chunk_emit):
        """Starts a line; `context` is the voice's reference frames ([T][16], or [] for none)."""
        self.jobs.put(("sync", None))
        self._wait("sync")                       # drops anything left of an aborted line
        self.pending = []
        self.emit_every = max(1, min(self.C, emit_every))
        self.first_emit = max(1, min(self.C, first_chunk_emit or emit_every))
        self.chunks = 0
        self.cold = len(context) < self.L        # an unprimed first window needs all 12 frames
        self.jobs.put(("begin", (context, key)))

    def push(self, frame):
        self.pending.append(frame)
        want = self.C if self.cold else (self.first_emit if self.chunks == 0 else self.emit_every)
        if len(self.pending) == want:
            self._submit()

    def ready(self):
        """Audio chunks decoded so far, without waiting."""
        chunks = []
        while True:
            try:
                kind, payload = self.results.get_nowait()
            except queue.Empty:
                return chunks
            self._handle(kind, payload, chunks)

    def finish(self):
        """Flushes the last partial chunk and waits for every chunk of the line."""
        if self.pending:
            self._submit()
        self.jobs.put(("end", None))
        return self._wait("end")

    def close(self):
        self.jobs.put(("quit", None))
        self.proc.join(timeout=5)

    def _submit(self):
        self.jobs.put(("chunk", np.array(self.pending, np.int64)))
        self.pending = []
        self.chunks += 1
        self.cold = False


# ======================
# Voice
# ======================
class Voice:
    """A voice as the talker sees it: the speaker embedding and, for the in-context
    clone, the reference codec frames with the reference text.

    It also caches, per language, the utterance-independent prefix of the talker
    prompt: role | codec tags, speaker, codec pad | reference text (+) reference
    codec. The talker's KV cache is written in place at each position, so once
    this prefix has been prefilled for a voice and language it stays valid in
    the cache: the next utterance of the same voice only prefills its own text
    positions after it (talker.prefix_key says what the cache holds).
    """

    def __init__(self, name, speaker_embed, frames=(), text=None, codec_icl=None):
        self.name = name
        self.speaker_embed = speaker_embed          # [1, 1, H]
        self.frames = list(frames)                  # reference codec frames (ICL) or []
        self.text = text                            # projected reference text [1, n_ref, H] (ICL) or None
        self.codec_icl = codec_icl                  # codec_bos + reference frames [1, n_icl, H] (ICL) or None
        self._prefix = {}

    @property
    def icl(self):
        return self.codec_icl is not None

    @property
    def need_tokens(self):
        """Tokens of the utterance that must be committed before the prefill."""
        return self.codec_icl.shape[1] - self.text.shape[1] if self.icl else 1

    def prefix(self, tts, language_id):
        """(embeddings [1, P, H], key) of the utterance-independent prompt prefix."""
        if language_id not in self._prefix:
            tc = tts.cfg["talker_config"]
            if language_id is None:
                tags = [tc["codec_nothink_id"], tc["codec_think_bos_id"], tc["codec_think_eos_id"]]
            else:
                tags = [tc["codec_think_id"], tc["codec_think_bos_id"], language_id, tc["codec_think_eos_id"]]
            codec_input = np.concatenate(
                [tts.codec_table[tags][None], self.speaker_embed,
                 tts.codec_table[[tc["codec_pad_id"], tc["codec_bos_id"]]][None]], axis=1)
            n = codec_input.shape[1]
            head = np.concatenate([np.repeat(tts.tts_pad, n - 2, axis=1), tts.tts_bos], axis=1) + codec_input[:, :-1]
            parts = [tts.role_embed, head]
            if self.icl:
                n_ref = self.text.shape[1]
                parts.append(self.text + self.codec_icl[:, :n_ref])     # reference text on its codec frames
            embeds = np.concatenate(parts, axis=1).astype(np.float32)
            self._prefix[language_id] = (embeds, codec_input[:, -1:], (self.name, language_id))
        return self._prefix[language_id]


# ======================
# Model
# ======================
class Qwen3TTSStreaming:
    def __init__(self, env_id):
        self.cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
        tc = self.cfg["talker_config"]
        self.hidden = tc["hidden_size"]
        self.num_groups = tc["num_code_groups"]                         # 16
        self.talker_vocab = tc["vocab_size"]                            # 3072
        self.group_vocab = tc["code_predictor_config"]["vocab_size"]    # 2048
        self.zero_row = self.talker_vocab + (self.num_groups - 1) * self.group_vocab
        self.rng = np.random.default_rng(args.seed)
        self.bench = Benchmark()
        self.encode, self.tokenizer = create_tokenizer()

        coreml_dir = os.path.join(args.coreml_dir, "coreml")
        host_dir = os.path.join(args.coreml_dir, "host")
        self.encoder = EncoderNet(env_id, self.bench)
        self.prompt = PromptNet(host_dir, self.bench)
        self.codec_table = CodecEmbedding(host_dir).table
        logger.info("loading the Core ML models (the first run compiles them for the Neural Engine, "
                    "which can take a minute or two)")
        self.talker = CoreMLTalker(coreml_dir, self.bench, self.hidden)
        self.code_predictor = CoreMLCodePredictor(coreml_dir, self.bench, self.rng, args.subtalker_temperature)
        self.decoder = CoreMLDecoder(coreml_dir, os.path.join(args.coreml_dir, "vochead"), self.bench,
                                     args.vocoder_units)
        self.providers = ["Core ML: talker / code predictor on the ANE, upsampler on {}".format(args.vocoder_units),
                          "encoder on {}".format("onnxruntime" if args.onnx else f"ailia env_id={env_id}")]

        # special text embeddings and the role prefix, one prompt call
        role = self.encode("<|im_start|>assistant\n")
        assert len(role) == 3, role
        ids = [self.cfg["tts_bos_token_id"], self.cfg["tts_eos_token_id"], self.cfg["tts_pad_token_id"]] + role
        proj = self.prompt.run(np.array([ids], dtype=np.int64))[0]
        self.tts_bos, self.tts_eos, self.tts_pad = proj[:, 0:1], proj[:, 1:2], proj[:, 2:3]
        self.role_embed = proj[:, 3:]

        eos_ids = {tc["codec_eos_token_id"], 2150, 2157, 151670, self.cfg["tts_eos_token_id"],
                   self.cfg["im_end_token_id"], 151643}
        self.eos_ids = eos_ids
        self.suppress = [i for i in range(self.talker_vocab - 1024, self.talker_vocab) if i not in eos_ids]
        self.codec_eos = [i for i in eos_ids if i < self.talker_vocab]
        self.clone_voice = self.voice_clone_prompt(args.ref_audio, args.ref_text)
        self.bench.reset()

    # ---------------- embeddings ----------------
    def group_embed(self, group, token):
        row = token if group == 0 else self.talker_vocab + (group - 1) * self.group_vocab + token
        return self.codec_table[row]

    def frame_embed(self, tokens):
        return sum(self.group_embed(g, t) for g, t in enumerate(tokens))

    def embed_committed(self, chunks, is_utterance_start=False):
        """Chunks from LiveTextSource carry their own leading space, so each is tokenized as is."""
        parts = []
        for i, c in enumerate(chunks):
            text = c.lstrip() if (is_utterance_start and i == 0) else c
            ids = self.encode(text) if text else []
            if ids:
                parts.append(self.prompt.run(np.array([ids], dtype=np.int64))[0])
        return np.concatenate(parts, axis=1) if parts else None

    # ---------------- voice clone ----------------
    def voice_clone_prompt(self, ref_audio, ref_text):
        """Encode the reference once: its codec frames, speaker embedding and the
        projected reference text, i.e. everything the ICL prompt is built from."""
        import librosa
        wav, sr = librosa.load(ref_audio, sr=SAMPLE_RATE, mono=True)
        codes, spk = self.encoder.run(wav[None, None].astype(np.float32), mel_spectrogram(wav, sr))
        frames = [[int(t) for t in f] for f in codes[0, : self.num_groups].T]   # [T][16]
        speaker = spk.reshape(1, 1, -1).astype(np.float32)
        if not ref_text.strip():
            # x_vector_only_mode of the reference implementation: the speaker embedding
            # takes the place of a CustomVoice speaker, the reference codes are unused
            logger.info("reference: {:.1f} s, speaker embedding only (no transcript)".format(len(wav) / sr))
            return Voice("clone:" + ref_audio, speaker)
        ids = self.encode(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n")
        text = self.prompt.run(np.array([ids[3:-2]], dtype=np.int64))[0]           # [1, n_ref_text, H]
        codec_icl = np.concatenate(
            [self.codec_table[[self.cfg["talker_config"]["codec_bos_id"]]][None],
             np.stack([self.frame_embed(f) for f in frames])[None]], axis=1)       # codec_bos + ref frames
        logger.info("reference: {:.1f} s, {} codec frames, {} text tokens -> the first {} tokens of each "
                    "utterance are needed before the prefill".format(
                        len(wav) / sr, len(frames), text.shape[1], codec_icl.shape[1] - text.shape[1]))
        return Voice("clone:" + ref_audio, speaker, frames, text, codec_icl)

    # ---------------- code predictor ----------------
    def predict_groups(self, token, past_hidden):
        """Groups 1..15 of the frame whose group 0 is `token`, and the frame's summed embedding."""
        groups = self.code_predictor.run(past_hidden, self.group_embed(0, token))
        return groups, self.frame_embed([token] + groups)

    # ---------------- the live text loop ----------------
    def stream(self, source, language,
               temperature=0.9, top_k=50,
               emit_every_frames=12, first_chunk_emit_every=0,
               wind_down_frames=200, wind_down_multiplier=6.0, max_frames=10000, on_event=None):
        """Yields PCM float32 chunks at SAMPLE_RATE. A numpy port of
        Qwen3-TTS-streaming-input's stream_generate_pcm_live_text:
        prefill once the first text has committed, then one talker step per frame,
        pausing (no forward) whenever the text has been caught up with, suppressing
        EOS until the source is closed. The frames go to the decoder thread, which
        hands back a chunk every emit_every_frames."""
        tc = self.cfg["talker_config"]
        ev = on_event or (lambda *a: None)

        # --- wait for the first committed text ---
        # Base reads text only where it is aligned with the reference codec frames
        # (trailing text alone is ignored), so the positions the reference text
        # leaves free have to be filled with the first tokens of this utterance
        # before the prefill.
        lang = (language or "auto").lower()
        language_id = None if lang == "auto" else tc["codec_language_id"][lang]
        voice = self.clone_voice
        first, closed = None, False
        while True:
            chunks, closed = source.poll()
            rows = self.embed_committed(chunks, is_utterance_start=first is None)
            if rows is not None:
                first = rows if first is None else np.concatenate([first, rows], axis=1)
            if (first is not None and first.shape[1] >= voice.need_tokens) or closed:
                break
            source.wait()
        if first is None:
            raise ValueError("the text source closed without any text")
        ev("first_commit")
        eos_appended = False

        # --- conditioning: [prefix: role | codec tags, speaker, codec pad | reference] + this utterance ---
        prefix, codec_bos, prefix_key = voice.prefix(self, language_id)
        if not voice.icl:
            # the speaker-embedding-only clone: the first text token sits on codec_bos, the rest is trailing
            utterance = first[:, :1] + codec_bos
            trailing = first[:, 1:]
        else:
            # ICL: this utterance's text (+ eos when complete) position by position on the
            # reference frames the reference text left free, padded with tts_pad if shorter;
            # what does not fit is consumed as trailing text
            n_ref = voice.text.shape[1]
            codec_rest = voice.codec_icl[:, n_ref:]
            text = first
            if closed:
                text = np.concatenate([text, self.tts_eos], axis=1)
                eos_appended = True
            n_rest = codec_rest.shape[1]
            if text.shape[1] < n_rest:
                text = np.concatenate([text, np.repeat(self.tts_pad, n_rest - text.shape[1], axis=1)], axis=1)
            utterance = text[:, :n_rest] + codec_rest
            trailing = text[:, n_rest:]
        utterance = utterance.astype(np.float32)

        def pick(logits, allow_eos):
            sup = self.suppress if allow_eos else self.suppress + self.codec_eos
            if allow_eos:
                # stop when EOS is the most likely token, not only when it is drawn: the fp16
                # talker on the ANE leaves EOS a little less probable, and a line that misses it
                # runs on in silence until the wind down (QwenANE's eosOnArgmax does the same)
                masked = logits.astype(np.float64, copy=True)
                masked[self.suppress] = -np.inf
                best = int(np.argmax(masked))
                if best in self.eos_ids:
                    return best
            return sample_token(logits, self.rng, temperature, top_k, 1.0, sup)

        # --- prefill: the prefix only when the cache does not hold it already ---
        prefill_len = prefix.shape[1] + utterance.shape[1]
        if prefill_len >= self.talker.buf:
            raise RuntimeError(f"prompt of {prefill_len} positions does not fit the talker's KV cache ({self.talker.buf})")
        if self.talker.prefix_key != prefix_key:
            self.talker.prefill(prefix)
            self.talker.prefix_key = prefix_key
            ev("prefix_prefill")
        logits, last_hidden = self.talker.prefill(utterance, start=prefix.shape[1])
        past_hidden = last_hidden
        position = prefill_len
        generation_step = 0
        token = pick(logits[0, -1], closed and generation_step >= trailing.shape[1])

        self.decoder.begin(voice.frames, voice.name, emit_every_frames, first_chunk_emit_every)
        frames_since_closed = real_frames = n_frames = 0
        ended = False

        # Once the whole text is in, the talker on the ANE often goes on in silence
        # instead of drawing EOS (seconds of it, up to the wind down). Silent audio
        # after the text is held back: speech that follows releases it, and
        # TAIL_SILENCE_S of it ends the utterance with KEEP_TAIL_S kept (QwenANE
        # stops on the trailing silence the same way).
        tail = {"text_done": False, "held": [], "held_s": 0.0}

        def emit(wavs):
            for wav in wavs:
                frame_rms = np.sqrt(np.mean(wav.reshape(-1, FRAME_SAMPLES) ** 2, axis=1))
                if tail["text_done"] and frame_rms.max() < SILENCE_RMS:
                    tail["held"].append(wav)
                    tail["held_s"] += len(wav) / SAMPLE_RATE
                    continue
                yield from tail["held"]
                tail["held"], tail["held_s"] = [], 0.0
                yield wav

        for _ in range(max_frames):
            g = generation_step
            new_chunks, closed = source.poll()
            new_rows = self.embed_committed(new_chunks)
            if closed and not eos_appended:
                new_rows = self.tts_eos if new_rows is None else np.concatenate([new_rows, self.tts_eos], axis=1)
                eos_appended = True
            # positions already passed with padding are backfilled so new text lands at g
            if trailing.shape[1] < g:
                trailing = np.concatenate([trailing, np.repeat(self.tts_pad, g - trailing.shape[1], axis=1)], axis=1)
            if new_rows is not None and new_rows.shape[1] > 0:
                trailing = np.concatenate([trailing, new_rows], axis=1)
            caught_up = trailing.shape[1] <= g

            frames_since_closed = frames_since_closed + 1 if (closed and caught_up) else 0
            if not caught_up:
                real_frames += 1
            if closed and frames_since_closed > max(wind_down_frames, int(wind_down_multiplier * real_frames)):
                ended = True
                break
            if not closed and caught_up:
                ev("pause")             # the talker has no clock: wait for text, then continue
                yield from emit(self.decoder.ready())
                source.wait()
                continue
            if token in self.eos_ids:
                ended = True
                break

            groups, frame_embed = self.predict_groups(token, past_hidden)
            self.decoder.push([token] + groups)
            n_frames += 1
            text_feedback = trailing[:, g:g + 1] if g < trailing.shape[1] else self.tts_pad
            x = (frame_embed[None, None] + text_feedback).astype(np.float32)
            if position >= self.talker.buf:
                raise RuntimeError(f"utterance longer than the talker's KV cache ({self.talker.buf} positions)")
            logits, last_hidden = self.talker.step(x, position)
            past_hidden = last_hidden
            position += 1
            generation_step += 1
            token = pick(logits[0, -1], closed and generation_step >= trailing.shape[1])
            ev("frame", n_frames)

            # --- emit the audio the decoder process has finished ---
            tail["text_done"] = closed and generation_step >= trailing.shape[1]
            yield from emit(self.decoder.ready())
            if tail["held_s"] >= TAIL_SILENCE_S:
                ended = True
                break

        # --- flush the frames not emitted yet, then the start of the held silence ---
        yield from emit(self.decoder.finish())
        if tail["held"]:
            yield np.concatenate(tail["held"])[:int(KEEP_TAIL_S * SAMPLE_RATE)]
        if not ended and not closed:
            raise RuntimeError(f"max_frames={max_frames} reached while text was still expected")


# ======================
# Drivers
# ======================
class Timeline:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.ev = {}

    def now(self):
        return time.perf_counter() - self.t0

    def mark(self, k):
        self.ev.setdefault(k, round(self.now(), 3))


class Player:
    """Plays chunks on the audio device (--play) and, either way, keeps a real-time
    clock to count underruns: moments when playback caught up with the audio made
    so far (the "gaps" of the evaluation reports)."""

    def __init__(self, tl, play=False, prebuffer_s=0.0):
        self.tl, self.pre = tl, prebuffer_s
        self.fed = 0.0; self.started = False; self.start = 0.0; self.underruns = 0; self.gap = 0.0
        self.stream = None; self.chunks = []; self.q = None
        if play:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            self.stream.start()
            # the device write blocks until its buffer has room, so it runs on its
            # own thread: generation must be free to run ahead of playback
            self.q = queue.Queue()
            self.thread = threading.Thread(target=self._play, daemon=True)
            self.thread.start()

    def _play(self):
        while True:
            chunk = self.q.get()
            if chunk is None:
                return
            self.stream.write(np.ascontiguousarray(chunk))

    def feed(self, chunk):
        self.chunks.append(chunk)
        now = self.tl.now()
        if self.started:
            played = now - self.start
            if played > self.fed:
                self.underruns += 1; self.gap += played - self.fed; self.start += played - self.fed
        self.fed += len(chunk) / SAMPLE_RATE
        if not self.started and self.fed >= self.pre:
            self.started = True; self.start = now; self.tl.mark("play_start")
        if self.q is not None:
            self.q.put(chunk)

    def close(self):
        if self.stream is not None:
            self.q.put(None); self.thread.join()      # let the queued audio play out
            self.stream.stop(); self.stream.close()
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def new_source(tts):
    if args.commit == "token":
        return LiveTextSource(tokenize=token_offsets_fn(tts.tokenizer), holdback_tokens=3)
    return LiveTextSource()


def run_one(tts, source, tl, label="", player=None):
    """Generate one utterance from `source`, play/collect it, and return (timings, audio).
    With a shared `player` the audio of consecutive utterances is one continuous stream."""
    pl = player or Player(tl, play=args.play, prebuffer_s=args.prebuffer)
    n0, u0, g0 = len(pl.chunks), pl.underruns, pl.gap
    frames = {"n": 0, "pauses": 0}

    def on_event(kind, *a):
        if kind == "first_commit":
            tl.mark("first_commit")
        elif kind == "pause":
            frames["pauses"] += 1
        elif kind == "prefix_prefill":
            tl.mark("prefix_prefill")        # the voice's prompt prefix was (re)filled this utterance
        elif kind == "frame":
            frames["n"] = a[0]
            tl.mark("first_frame")

    err = None
    try:
        for chunk in tts.stream(source, args.language,
                                temperature=args.temperature, top_k=args.top_k,
                                emit_every_frames=args.emit_every,
                                first_chunk_emit_every=args.first_chunk_emit, on_event=on_event):
            tl.mark("first_pcm")
            pl.feed(chunk)
    except Exception as e:  # noqa: BLE001 -- report, keep the session alive
        err = repr(e)
        logger.error(err)
    tl.mark("gen_done")
    if player is None:
        audio = pl.close()
    else:
        audio = np.concatenate(pl.chunks[n0:]) if len(pl.chunks) > n0 else np.zeros(0, np.float32)
    dur = len(audio) / SAMPLE_RATE
    gen = tl.ev.get("gen_done", 0) - tl.ev.get("first_pcm", 0)
    r = dict(label=label, audio_s=round(dur, 2), frames=frames["n"], pauses=frames["pauses"],
             rtf=round(gen / dur, 2) if dur else None, underruns=pl.underruns - u0, gap_s=round(pl.gap - g0, 2),
             ev=tl.ev, error=err)
    # with a shared player the text being echoed owns the console: per utterance lines go to --debug
    (logger.debug if player is not None else logger.info)(json.dumps(r, ensure_ascii=False))
    return r, audio


def split_utterances(text, max_chars):
    """Utterances: whole sentences (ending in 。！？ or at a line end) packed up to
    max_chars; a sentence longer than that is cut at 、 or, failing that, at max_chars."""
    sentences = []
    for line in text.split("\n"):
        cur = ""
        for ch in line:
            cur += ch
            if ch in UTTERANCE_END:
                sentences.append(cur); cur = ""
        if cur.strip():
            sentences.append(cur)
    out, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:
            cut = s.rfind("、", 0, max_chars) + 1 or max_chars
            if cur:
                out.append(cur); cur = ""
            out.append(s[:cut]); s = s[cut:]
        if cur and len(cur) + len(s) > max_chars:
            out.append(cur); cur = ""
        cur += s
    if cur.strip():
        out.append(cur)
    return out


def speak_text(tts, text):
    """Feed `text` in pieces at --cps, echo each piece as it goes in, and speak it as
    it arrives: one LiveTextSource per utterance, generated back to back into one
    Player, while the producer thread keeps feeding the following utterances."""
    utterances = split_utterances(text, args.max_chars)
    sources = queue.Queue()
    tl = Timeline()
    pl = Player(tl, play=args.play, prebuffer_s=args.prebuffer)
    echo = (lambda s: None) if args.quiet else (lambda s: (sys.stdout.write(s), sys.stdout.flush()))

    def prod():
        for u in utterances:
            src = new_source(tts); sources.put((src, Timeline()))
            for i in range(0, len(u), args.piece):
                src.push(u[i:i + args.piece]); echo(u[i:i + args.piece]); time.sleep(args.piece / args.cps)
            src.close(); echo("\n")
        sources.put(None)
    threading.Thread(target=prod, daemon=True).start()

    results = []
    while True:
        item = sources.get()
        if item is None:
            break
        src, utl = item
        r, _ = run_one(tts, src, utl, label=f"utterance{len(results)}", player=pl)
        results.append(r)
    audio = pl.close()
    logger.info("spoke {} utterances, {:.1f} s of audio in {:.1f} s, play start {:.2f} s, underruns {} ({:.2f} s)".format(
        len(results), len(audio) / SAMPLE_RATE, tl.now(), tl.ev.get("play_start", 0), pl.underruns, pl.gap))
    return results, audio


def download_coreml():
    """The Core ML build from Hugging Face (models, host tables, vocoder head; not its encoders)."""
    host_cfg = os.path.join(args.coreml_dir, "host", "config.json")
    if os.path.exists(host_cfg) and os.path.isdir(os.path.join(args.coreml_dir, "coreml", "talker1.mlmodelc")):
        return
    from huggingface_hub import snapshot_download
    logger.info(f"downloading {COREML_REPO} to {args.coreml_dir} (about 1.7 GB)")
    snapshot_download(COREML_REPO, local_dir=args.coreml_dir, allow_patterns=COREML_FILES)


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_ENCODER, MODEL_PATH_ENCODER, REMOTE_PATH)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    check_and_download_file(VOCAB_PATH, TOKENIZER_REMOTE_PATH)
    check_and_download_file(MERGES_PATH, TOKENIZER_REMOTE_PATH)
    download_coreml()

    t = time.perf_counter()
    tts = Qwen3TTSStreaming(args.env_id)
    logger.info("loaded in {:.1f}s: {}".format(time.perf_counter() - t, ", ".join(map(str, tts.providers))))
    logger.info(f"model: 0.6B base (voice clone of {args.ref_audio})")
    logger.info("talker: KV cache of {} positions".format(tts.talker.buf))

    # warm up (the first calls of every Core ML function are slow)
    s = new_source(tts); s.push("こんにちは、テストです。"); s.close()
    for _ in tts.stream(s, args.language):
        pass
    tts.bench.reset()

    results, audios = [], []
    inputs = args.input if isinstance(args.input, list) else [args.input]
    for inp in inputs:
        # a text file, or the text itself
        text = open(inp, encoding="utf-8").read() if os.path.isfile(inp) else inp
        logger.info(f"speaking {inp} ({len(text)} characters) fed at {args.cps:g} characters/s")
        rs, audio = speak_text(tts, text)
        results += rs; audios.append(audio)

    if args.benchmark:
        tts.bench.report()
    tts.decoder.close()

    import soundfile as sf
    base, ext = os.path.splitext(args.savepath)
    for i, a in enumerate(audios):
        path = f"{base}_{i}{ext}" if len(audios) > 1 else args.savepath
        sf.write(path, a, SAMPLE_RATE)
        logger.info(f"Saved as {path}.")
    if args.results:
        with open(args.results, "a", encoding="utf-8") as f:
            for r in results:
                r.update(runtime="coreml", parameter_num="0.6B", model="base", commit=args.commit,
                         vocoder_units=args.vocoder_units, emit_every=args.emit_every, timing=tts.bench.as_dict())
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
