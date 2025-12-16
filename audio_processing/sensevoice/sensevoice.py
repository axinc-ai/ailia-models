#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from pathlib import Path
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

from logging import getLogger
logger = getLogger(__name__)

model_dir = "iic/SenseVoiceSmall"

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False, cache_dir="./")

# inference
wav_or_scp = ["./example/ax.wav"]
#wav_or_scp = ["./example/en.mp3", "./example/ja.mp3", "./example/ax.wav"]

import time

start = int(round(time.time() * 1000))
res = model(wav_or_scp, language="auto", use_itn=True)
end = int(round(time.time() * 1000))
estimation_time = end - start
#logger.info(f"\tencoder processing time {estimation_time} ms")
print(f"\tprocessing time {estimation_time} ms")

print([rich_transcription_postprocess(i) for i in res])
