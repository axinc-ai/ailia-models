import numpy as np
from threading import Lock

import os
import ailia
import psutil
import math

from sentencepiece import SentencePieceProcessor
from typing import List

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

class OrtWrapper:
    def __init__(self, onnxfile: str, env_id: int):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=True)
        self.sess = ailia.Net(None, onnxfile, env_id = env_id, memory_mode = memory_mode)
        #print(onnxfile)

    def forward(self, _inputs: dict):
        output_tensors = self.sess.run(_inputs)

        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[i] = tensor

        return output
    
class MemoryPoolSimple:
    def __init__(self, maxGB, env_id):
        if maxGB < 0:
            raise Exception('maxGB must > 0, get {}'.format(maxGB))
        
        self.max_size = maxGB * 1024 * 1024 * 1024
        self.wait_map = {}
        self.active_map = {}
        self.env_id = env_id

    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'file_size': os.path.getsize(onnx_filepath)
            }

    def used(self):
        sum_size = 0
        biggest_k = None
        biggest_size = 0
        for k in self.active_map.keys():
            cur_size = self.wait_map[k]['file_size']
            sum_size += cur_size

            if biggest_k is None:
                biggest_k = k
                biggest_size = cur_size
                continue
            
            if cur_size > biggest_size:
                biggest_size = cur_size
                biggest_k = k
        
        return sum_size, biggest_k

    def check(self):
        sum_need = 0
        for k in self.wait_map.keys():
            sum_need = sum_need + self.wait_map[k]['file_size']
            
        sum_need /= (1024 * 1024 * 1024)
        
        total = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        if total > 0 and total < sum_need:
            logger.warning('virtual_memory not enough, require {}, try `--poolsize {}`'.format(sum_need, math.floor(total)))


    def fetch(self, key: str):
        if key in self.active_map:
            return self.active_map[key]
        
        need = self.wait_map[key]['file_size']
        onnx = self.wait_map[key]['onnx']

        # check current memory use
        used_size, biggest_k = self.used()
        while biggest_k is not None and self.max_size - used_size < need:
            # if exceeded once, delete until `max(half_max, file_size)` left
            need = max(need, self.max_size / 2)
            if len(self.active_map) == 0:
                break

            del self.active_map[biggest_k]
            used_size, biggest_k = self.used()
        
        self.active_map[key] = OrtWrapper(onnx, env_id = self.env_id)
        return self.active_map[key]

# refers to https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py 
def warp_topk(tensor: np.array, topk: int, fill_value = -float("Inf")):
    if topk is None or topk <= 0:
        return tensor
    assert len(tensor.shape) == 2
    
    if topk > tensor.shape[-1]:
        logger.warning('topk value {} bigger than tensor shape {}, updated'.format(topk, tensor.shape))
        topk = min(topk, tensor.shape[-1])

    for idx, pval in enumerate(tensor):
        # for each row, loop
        non_topk_idx = np.argpartition(pval, -topk)[0:-topk]
        tensor[idx][non_topk_idx] = fill_value

    return tensor


def warp_temperature(tensor: np.array, temperature: float):
    EPSILON = 1e-4
    if abs(temperature - 1.0) <= EPSILON:
        return tensor
    
    if temperature <= EPSILON:
        raise Exception('bad temperature {}, make sure `0.0 < temperature < 1.0`')
    
    return tensor / temperature

# copy from github.com/BLinkDL/ChatRWKV
def sample_logits(probs, temperature=1.0, top_p=0.85):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out


def singleton(cls):
    _instance = {}
    _instance_lock = Lock()

    def inner(*args, **kwargs):
        if cls not in _instance:
            with _instance_lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


def npsoftmax(x, axis):
    y = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)


def npmultinominal2D(x):
    assert len(x.shape) == 2

    ret = np.zeros((x.shape[0], 1), dtype=x.dtype)

    for row, pval in enumerate(x):
        ret[row] = np.random.multinomial(1, pval).argmax()

    return ret


class Decoder:
    def __init__(self, pool: MemoryPoolSimple, onnxdir: str, nameformat: str, count: int = 32):

        # reload tokenizer
        assert os.path.isdir(onnxdir)
        self._pool = pool

        for idx in range(count):
            filepath = os.path.join(onnxdir, nameformat.format(idx))
            self._pool.submit('decode{}'.format(idx),filepath)

        self._pool.submit('embed', os.path.join(onnxdir, 'embed.onnx'))
        self._pool.submit('norm', os.path.join(onnxdir, 'norm.onnx'))
        self._pool.submit('head', os.path.join(onnxdir, 'head.onnx'))

    def decode(self, _inputs: dict, idx: int):
        key = 'decode{}'.format(idx)

        handler = self._pool.fetch(key)
        outputs = handler.forward(_inputs)
        
        return outputs

    def embed(self, input_ids: np.array):
        handler = self._pool.fetch('embed')

        input_embed = handler.forward({'input': input_ids})[0]
        return input_embed

    def norm_head(self, hidden: np.array):
        handler = self._pool.fetch('norm')
        output = handler.forward({'input': hidden})[0]

        handler = self._pool.fetch('head')
        output = handler.forward({'input': output})[0]
        return output


class Tokenizer:

    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
