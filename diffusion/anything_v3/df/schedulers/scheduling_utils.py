# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import numpy as np

from ..utils import BaseOutput


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: np.ndarray


class SchedulerMixin:
    """
    Mixin containing common functions for the schedulers.

    Class attributes:
        - **_compatibles** (`List[str]`) -- A list of classes that are compatible with the parent class, so that
          `from_config` can be used from a class different than the one used to save the config (should be overridden
          by parent class).
    """

    config_name = SCHEDULER_CONFIG_NAME
    _compatibles = []
    has_compatibles = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Dict[str, Any] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        r"""
        Instantiate a Scheduler class from a pre-defined JSON configuration file inside a directory or Hub repo.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing the schedluer configurations saved using
                      [`~SchedulerMixin.save_pretrained`], e.g., `./my_model_directory/`.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        """
        config, kwargs = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            **kwargs,
        )
        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    @property
    def compatibles(self):
        """
        Returns all schedulers that are compatible with this scheduler

        Returns:
            `List[SchedulerMixin]`: List of compatible schedulers
        """
        return self._get_compatibles()

    @classmethod
    def _get_compatibles(cls):
        compatible_classes_str = list(set([cls.__name__] + cls._compatibles))
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        compatible_classes = [
            getattr(diffusers_library, c) for c in compatible_classes_str if hasattr(diffusers_library, c)
        ]
        return compatible_classes
