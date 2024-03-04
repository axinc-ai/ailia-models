import yaml
import os
import pathlib

from pyannote.audio import Pipeline
from pyannote.core import Segment, notebook, Annotation
import json
# import polars as pl
import torch
from pyannote.core import Segment, notebook
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
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
    with open("./model/config.yaml", 'r') as yml:
        config = yaml.safe_load(yml)

    config["pipeline"]["params"]["segmentation"] = "./model/segmentation.onnx"
    config["pipeline"]["params"]["embedding"] = "./model/speaker-embedding.onnx"
    with open("./model/config.yaml", 'w') as f:
        yaml.dump(config, f)

    # os.environ['HUGGINGFACE_HUB_CACHE'] = str(pathlib.Path("./tmp/assets").absolute())
    
    audio_file = args.file
    pipeline = Pipeline.from_pretrained("./model/config.yaml")

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
