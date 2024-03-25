import yaml
import sys

from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)


WEIGHT_SEGMENTATION_PATH = 'segmentation.onnx'
# MODEL_VOX_PATH = 'voxceleb_resnet34.onnx.prototxt'
WEIGHT_EMBEDDING_PATH = 'speaker-embedding.onnx'
# MODEL_CNC_PATH = 'cnceleb_resnet34.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pyannote-audio/'
YAML_PATH = 'config.yaml'

parser = get_base_parser(
    'Pyannote-audio', None, None, input_ftype='audio'
)

parser.add_argument(
    '--num', '-num_speaker', default=0, type=int,
    help='If the number of speakers is fixed', 
)
parser.add_argument(
    '--max', '-max_speaker', default=0, type=int,
    help='If the maximum number of speakers is fixed', 
)
parser.add_argument(
    '--min', '-min_speaker', default=0, type=int,
    help='If the minimum number of speakers is fixed', 
)
parser.add_argument(
    '--insu', '-input_sample', default='./data/sample.wav',
    help='Specify an input wav file'
) 
parser.add_argument(
    '--gr', '-ground', default='./data/sample.rttm',
    help='Specify a wav file as ground truth'
) 
parser.add_argument(
    '--embed', 
    action='store_true',
    help='When you need embedding vector', 
) 
parser.add_argument(
    '--use_onnx', 
    action='store_true',
    help='execute onnxruntime version.'
)

args = update_parser(parser)

def main(args):
    check_and_download_models(WEIGHT_SEGMENTATION_PATH, model_path=None,remote_path=REMOTE_PATH)
    check_and_download_models(WEIGHT_EMBEDDING_PATH, model_path=None,remote_path=REMOTE_PATH)
    
    with open(YAML_PATH, 'r') as yml:
        config = yaml.safe_load(yml)

    config["pipeline"]["params"]["segmentation"] = WEIGHT_SEGMENTATION_PATH
    config["pipeline"]["params"]["embedding"] = WEIGHT_EMBEDDING_PATH
    with open(YAML_PATH, 'w') as f:
        yaml.dump(config, f)

    
    audio_file = args.insu

    checkpoint_path = YAML_PATH
    config_yml = checkpoint_path
    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    params = config["pipeline"].get("params", {})
    pipeline = SpeakerDiarization(**params, args=args)
    if "params" in config:
        pipeline.instantiate(config["params"])
    
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
            
    _, groundtruth = load_rttm(args.gr).popitem()
    metric = DiarizationErrorRate()
    result = metric(groundtruth, diarization, detailed=False)
    print(diarization)
    print('Diarization error rate : ' + str(result))
    

if __name__ == "__main__":
    main(args)
