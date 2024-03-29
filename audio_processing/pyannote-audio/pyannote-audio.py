import yaml
import sys
import matplotlib.pyplot as plt

from pyannote_audio_utils.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote_audio_utils.core import Segment, Annotation
from pyannote_audio_utils.core.notebook import Notebook
from pyannote_audio_utils.database.util import load_rttm
from pyannote_audio_utils.metrics.diarization import DiarizationErrorRate

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)

WEIGHT_SEGMENTATION_PATH = 'segmentation.onnx'
MODEL_SEGMENTATION_PATH = 'segmentation.onnx.prototxt'
WEIGHT_EMBEDDING_PATH = 'speaker-embedding.onnx'
MODEL_EMBEDDING_PATH = 'speaker-embedding.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/pyannote-audio/'
YAML_PATH = 'config.yaml'
OUT_PATH = 'output.png'

parser = get_base_parser(
    'Pyannote-audio', './data/sample.wav', None, input_ftype='audio'
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
    '--ig', '-ground', default=None,
    help='Specify a wav file as ground truth. If you need diarization error rate, you need this file'
)
parser.add_argument(
    '--o', '-output', default='output.png',
    help='Specify an output file'
) 
parser.add_argument(
    '--og', '-output_ground', default='output_ground.png',
    help='Specify an output ground truth file'
) 
parser.add_argument(
    '--e', '-error',
    action='store_true',
    help='If you need diarization error rate'
)
parser.add_argument(
    '--plt',
    action='store_true',
    help='If you want to visualize result'
)
parser.add_argument(
    '--embed', 
    action='store_true',
    help='If you need embedding vector', 
) 
parser.add_argument(
    '--use_onnx', 
    action='store_true',
    help='execute onnxruntime version'
)

args = update_parser(parser)

def repr_annotation(args, annotation: Annotation, notebook:Notebook, ground:bool = False):
    """Get `png` data for `annotation`"""
    figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_annotation(annotation, ax=ax)
    if ground:
        plt.savefig(args.og)
    else:
        plt.savefig(args.o)
    plt.close(fig)
    plt.rcParams["figure.figsize"] = figsize
    return

def main(args):
    check_and_download_models(WEIGHT_SEGMENTATION_PATH, MODEL_SEGMENTATION_PATH, remote_path=REMOTE_PATH)
    check_and_download_models(WEIGHT_EMBEDDING_PATH, MODEL_EMBEDDING_PATH, remote_path=REMOTE_PATH)
    
    with open(YAML_PATH, 'r') as yml:
        config = yaml.safe_load(yml)

    config["pipeline"]["params"]["segmentation"] = WEIGHT_SEGMENTATION_PATH
    config["pipeline"]["params"]["embedding"] = WEIGHT_EMBEDDING_PATH
    with open(YAML_PATH, 'w') as f:
        yaml.dump(config, f)

    audio_file = args.input[0]
    checkpoint_path = YAML_PATH
    config_yml = checkpoint_path

    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    params = config["pipeline"].get("params", {})
    pipeline = SpeakerDiarization(
        **params,
        args=args, 
        seg_path=MODEL_SEGMENTATION_PATH, 
        emb_path=MODEL_EMBEDDING_PATH,
    )

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
    
    if args.ig:
        _, groundtruth = load_rttm(args.ig).popitem()
        metric = DiarizationErrorRate()
        result = metric(groundtruth, diarization, detailed=False)
        
        mapping = metric.optimal_mapping(groundtruth, diarization)
        diarization = diarization.rename_labels(mapping=mapping)

        print(diarization)
        if args.e:
            print(f'diarization error rate = {100 * result:.1f}%')

        if args.plt:
            EXCERPT = Segment(0, 30)
            notebook = Notebook()
            notebook.crop = EXCERPT
            repr_annotation(args, diarization, notebook)
            repr_annotation(args, groundtruth, notebook, ground=True)
        return
    
    else:
        print(diarization)

        if args.plt:
            EXCERPT = Segment(0, 30)
            notebook = Notebook()
            notebook.crop = EXCERPT
            repr_annotation(args, diarization, notebook)
        return 


if __name__ == "__main__":
    main(args)
