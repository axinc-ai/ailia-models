# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Dict, List,  Union

import numpy as np
import yaml

import warnings

root_dir = Path(__file__).resolve().parent

class OrtInferSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        from onnxruntime import (
            GraphOptimizationLevel,
            InferenceSession,
            SessionOptions,
            get_available_providers,
            get_device,
        )

        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if device_id != "-1" and get_device() == "GPU" and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(model_file)
        self.session = InferenceSession(model_file, sess_options=sess_opt, providers=EP_list)

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(self, input_content: List[Union[np.ndarray, np.ndarray]], run_options = None) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(self.get_output_names(), input_dict, run_options)
        except Exception as e:
            raise Exception("ONNXRuntime inferece failed.")

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")



class AiliaInferSession:
    def __init__(self, model_file, env_id = -1):
        import ailia
        self.session = ailia.Net(weight=model_file, env_id=env_id, memory_mode=11)

    def __call__(self, input_content: List[Union[np.ndarray, np.ndarray]], run_options = None) -> np.ndarray:
        return self.session.run(input_content)

def split_to_mini_sentence(words: list, word_limit: int = 20):
    assert word_limit > 1
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit : (i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit :])
    return sentences


def code_mix_split_words(text: str):
    words = []
    segs = text.split()
    for seg in segs:
        # There is no space in seg.
        current_word = ""
        for c in seg:
            if len(c.encode()) == 1:
                # This is an ASCII char.
                current_word += c
            else:
                # This is a Chinese char.
                if len(current_word) > 0:
                    words.append(current_word)
                    current_word = ""
                words.append(c)
        if len(current_word) > 0:
            words.append(current_word)
    return words


def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
