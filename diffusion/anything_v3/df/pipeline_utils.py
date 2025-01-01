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

import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

#import diffusers
import PIL
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

from .configuration_utils import ConfigMixin
#from .dynamic_modules_utils import get_class_from_dynamic_module
#from .hub_utils import http_user_agent
from .utils import (
    BaseOutput,
)



LOADABLE_CLASSES = {
    "diffusers": {
        "SchedulerMixin": ["from_pretrained"],
        "DiffusionPipeline": ["from_pretrained"],
        "OnnxRuntimeModel": ["from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised samples of shape `(batch_size, num_channels, sample_rate)`. Numpy array present the
            denoised audio samples of the diffusion pipeline.
    """

    audios: np.ndarray


def is_safetensors_compatible(info) -> bool:
    filenames = set(sibling.rfilename for sibling in info.siblings)
    pt_filenames = set(filename for filename in filenames if filename.endswith(".bin"))
    is_safetensors_compatible = any(file.endswith(".safetensors") for file in filenames)
    for pt_filename in pt_filenames:
        prefix, raw = os.path.split(pt_filename)
        if raw == "pytorch_model.bin":
            # transformers specific
            sf_filename = os.path.join(prefix, "model.safetensors")
        else:
            sf_filename = pt_filename[: -len(".bin")] + ".safetensors"
        if is_safetensors_compatible and sf_filename not in filenames:
            #logger.warning(f"{sf_filename} not found")
            is_safetensors_compatible = False
    return is_safetensors_compatible


class DiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`DiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion pipelines
    and handles methods for loading, downloading and saving models as well as a few methods common to all pipelines to:

        - move all PyTorch modules to the device of your choice
        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
        - **_optional_components** (List[`str`]) -- list of all components that are optional so they don't have to be
          passed for the pipeline to function (should be overridden by subclasses).
    """
    config_name = "model_index.json"
    _optional_components = []

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        #from diffusers import pipelines
        #import pipelines


        for name, module in kwargs.items():
            # retrieve library
            if module is None:
                register_dict = {name: (None, None)}
            else:
                library = module.__module__.split(".")[0]

                # check if the module is a pipeline module
                pipeline_dir = module.__module__.split(".")[-2] if len(module.__module__.split(".")) > 2 else None
                path = module.__module__.split(".")
                
                #is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                #print("is_pipeline_module: ", is_pipeline_module, name)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                #if library not in LOADABLE_CLASSES or is_pipeline_module:
                #    library = pipeline_dir
                

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
