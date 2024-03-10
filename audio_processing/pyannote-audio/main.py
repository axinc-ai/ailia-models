import yaml
import os
import pathlib
import sys
import time 
import numpy as np
import librosa
# import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from importlib import import_module
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import json
# import polars as pl
import torch
from pyannote.core import Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

WEIGHT_SEGMENTATION_PATH = 'segmentation.onnx'
# MODEL_VOX_PATH = 'voxceleb_resnet34.onnx.prototxt'
WEIGHT_EMBEDDING_PATH = 'speaker-embedding.onnx'
# MODEL_CNC_PATH = 'cnceleb_resnet34.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pyannote-audio/'
YAML_PATH = './model/config.yaml'
import argparse    
parser = argparse.ArgumentParser(description='引数')   

parser.add_argument('--num','-num_speaker', default=0,help='話者数が決まっている場合', type=int)
parser.add_argument('--max','-max_speaker', default=0,help='話者数の最大数が決まっている場合', type=int)
parser.add_argument('--min','-min_speaker', default=0,help='話者数の最小数が決まっている場合', type=int)
parser.add_argument('--file', default='./data/demo.wav',help='解析するデータファイルの場所') 
parser.add_argument('--gfile', default='./data/demo_ground.rttm',help='解析するデータファイルの場所') 
parser.add_argument('--embed', default=False,help='話者の埋め込みベクトルが必要な場合', type=bool) 
args = parser.parse_args()   



def main(args):
    check_and_download_models(WEIGHT_SEGMENTATION_PATH, model_path=None,remote_path=REMOTE_PATH)
    check_and_download_models(WEIGHT_EMBEDDING_PATH, model_path=None,remote_path=REMOTE_PATH)
    
    with open(YAML_PATH, 'r') as yml:
        config = yaml.safe_load(yml)

    config["pipeline"]["params"]["segmentation"] = WEIGHT_SEGMENTATION_PATH
    config["pipeline"]["params"]["embedding"] = WEIGHT_EMBEDDING_PATH
    with open(YAML_PATH, 'w') as f:
        yaml.dump(config, f)

    # os.environ['HUGGINGFACE_HUB_CACHE'] = str(pathlib.Path("./tmp/assets").absolute())
    audio_file = args.file
    
    checkpoint_path = YAML_PATH
    config_yml = checkpoint_path
    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    params = config["pipeline"].get("params", {})
    # pipeline = Klass(**params)
    pipeline = SpeakerDiarization(**params)
    if "params" in config:
        pipeline.instantiate(config["params"])

    # send pipeline to GPU (when available)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    else:
        pipeline.to(torch.device("cpu"))
    
    if args.embed:
        if args.num > 0:
            diarization, embeddings = pipeline(audio_file, return_embeddings=True, num_speakers=args.num)
            for s, speaker in enumerate(diarization.labels()):
                print(speaker, embeddings[s].shape)
        elif args.max > 0 or args.min > 0:
            diarization, embeddings = pipeline(audio_file, return_embeddings=True, min_speakers=args.min, max_speaker=args.max)
            for s, speaker in enumerate(diarization.labels()):
                print(speaker, embeddings[s].shape)
        else:
            diarization, embeddings = pipeline(audio_file, return_embeddings=True)
            for s, speaker in enumerate(diarization.labels()):
                print(speaker, embeddings[s].shape)
    else:
        if args.num > 0:
            diarization = pipeline(audio_file, num_speakers=args.num)
        elif args.max > 0 or args.min > 0:
            diarization = pipeline(audio_file, min_speakers=args.min, max_speaker=args.max)
        else:
            diarization = pipeline(audio_file)
            
    _, groundtruth = load_rttm(args.gfile).popitem()

    metric = DiarizationErrorRate()
    result = metric(groundtruth, diarization, detailed=True)
    print(result)
    breakpoint()

if __name__ == "__main__":
    main(args)
