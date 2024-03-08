# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

#from huggingface_hub import hf_hub_download

#from .utils import ONNX_EXTERNAL_WEIGHTS_NAME, ONNX_WEIGHTS_NAME, is_onnx_available, logging
ONNX_WEIGHTS_NAME = "model.onnx"


#logger = logging.get_logger(__name__)

ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}


class OnnxRuntimeModel:
    def __init__(self, model=None, onnx=False, **kwargs):
        #logger.info("`diffusers.OnnxRuntimeModel` is experimental and might change in the future.")
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.onnx = onnx
        #self.latest_model_name = kwargs.get("latest_model_name", ONNX_WEIGHTS_NAME)

    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        if not self.onnx:
            return self.model.run(inputs)
        return self.model.run(None, inputs)

    @staticmethod
    def load_model(path: Union[str, Path], onnx=False, env_id=-1, provider=None, sess_options=None):
        """
        Loads an ONNX Inference session with an ExecutionProvider. Default provider is `CPUExecutionProvider`

        Arguments:
            path (`str` or `Path`):
                Directory from which to load
            provider(`str`, *optional*):
                Onnxruntime execution provider to use for loading the model, defaults to `CPUExecutionProvider`
        """
        if not onnx:
            import ailia
            memory_mode = ailia.get_memory_mode(
                reduce_constant=True, ignore_input_with_initializer=True,
                reduce_interstage=False, reuse_interstage=True)
            return ailia.Net(weight = path, env_id = env_id, memory_mode = memory_mode)

        if provider is None:
            #logger.info("No onnxruntime provider specified, using CPUExecutionProvider")
            provider = "CPUExecutionProvider"

        import onnxruntime as ort
        return ort.InferenceSession(path, providers=[provider], sess_options=sess_options)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        onnx: bool = False,
        env_id: int = -1,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        provider: Optional[str] = None,
        sess_options: Optional["ort.SessionOptions"] = None,
        **kwargs,
    ):
        """
        Load a model from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private or gated repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            provider(`str`):
                The ONNX runtime provider, e.g. `CPUExecutionProvider` or `CUDAExecutionProvider`.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
        # load model from local directory
        if os.path.isdir(model_id):
            model = OnnxRuntimeModel.load_model(
                os.path.join(model_id, model_file_name), onnx, env_id, provider=provider, sess_options=sess_options
            )
            kwargs["model_save_dir"] = Path(model_id)
        # load model from hub
        else:
            exit()
            """
            # download model
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=model_file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name
            model = OnnxRuntimeModel.load_model(model_cache_path, provider=provider, sess_options=sess_options)
            """
        return cls(model=model, onnx=onnx, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        file_name: Optional[str],
        onnx: bool = False,
        env_id: int = -1,
        force_download: bool = True,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs,
    ):
        revision = None
        if len(str(model_id).split("@")) == 2:
            model_id, revision = model_id.split("@")

        return cls._from_pretrained(
            model_id=model_id,
            onnx=onnx,
            env_id=env_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            use_auth_token=use_auth_token,
            file_name=file_name,
            **model_kwargs,
        )
