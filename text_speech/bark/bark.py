import sys
import time
import re
from logging import getLogger

import tqdm
import numpy as np
from transformers import BertTokenizer

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'xxx.onnx'
MODEL_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/bark/'

SAVE_WAV_PATH = 'output.wav'

SEMANTIC_VOCAB_SIZE = 10_000

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Bark', None, SAVE_WAV_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="""
    Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
    But I also have other interests such as playing tic tac toe.
    """,
    help="the prompt to render"
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


# ======================
# Main functions
# ======================

def generate_text_semantic(
        models,
        text,
        temp=0.7,
        top_k=None,
        top_p=None,
        silent=False,
        min_eos_p=0.2,
        max_gen_duration_s=None,
        allow_early_stop=True):
    """Generate semantic tokens from text."""
    text = normalize_whitespace(text)
    assert len(text.strip()) > 0

    tokenizer = models["tokenizer"]
    encoded_text = np.array(tokenizer.encode(text, add_special_tokens=False))
    encoded_text = encoded_text + TEXT_ENCODING_OFFSET

    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]

    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )

    semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = np.hstack([
        encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
    ]).astype(np.int64)
    x = np.expand_dims(x, axis=0)
    assert x.shape[1] == 256 + 256 + 1

    net = models["net"]
    offset = 0
    kv_cache = np.zeros((48, 16, 1024, 64), dtype=np.float32)

    pbar_state = 0
    tot_generated_duration_s = 0
    n_tot_steps = 768
    pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
    for n in range(n_tot_steps):
        if n == 0:
            x_input = x
        else:
            x_input = x[:, [-1]]
            offset = offset + 1

        # feedforward
        if not args.onnx:
            output = net.predict([
                x_input, kv_cache, np.array(offset, dtype=np.int64)
            ])
        else:
            output = net.run(None, {
                'x_input': x_input, 'past_kv': kv_cache,
                'offset': np.array(offset, dtype=np.int64)
            })
        logits, kv_cache = output

        relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
        if allow_early_stop:
            relevant_logits = torch.hstack(
                (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
            )
        if top_p is not None:
            # faster to convert to numpy
            original_device = relevant_logits.device
            relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
            sorted_indices = np.argsort(relevant_logits)[::-1]
            sorted_logits = relevant_logits[sorted_indices]
            cumulative_probs = np.cumsum(softmax(sorted_logits))
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
            relevant_logits = torch.from_numpy(relevant_logits)
            relevant_logits = relevant_logits.to(original_device)
        if top_k is not None:
            v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
            relevant_logits[relevant_logits < v[-1]] = -float("Inf")
        probs = F.softmax(relevant_logits / temp, dim=-1)
        item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
        if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
        ):
            # eos found, so break
            pbar.update(n - pbar_state)
            break
        x = torch.cat((x, item_next[None]), dim=1)
        tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
        if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
            pbar.update(n - pbar_state)
            break
        if n == n_tot_steps - 1:
            pbar.update(n - pbar_state)
            break
        del logits, relevant_logits, probs, item_next

        if n > pbar_state:
            if n > pbar.total:
                pbar.total = n
            pbar.update(n - pbar_state)
        pbar_state = n

    pbar.total = n
    pbar.refresh()
    pbar.close()
    out = x.squeeze()[256 + 256 + 1:]

    return out


def semantic_to_waveform(
        semantic_tokens: np.ndarray,
        temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    history_prompt = None
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    print("generate_fine---")
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    print("codec_decode---")
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def generate_audio(
        models,
        text: str,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        silent: bool = False,
        output_full: bool = False):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """

    x_semantic = generate_text_semantic(
        models,
        text,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        x_semantic,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )

    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out

    return audio_arr


def recognize_from_text(models):
    text_prompt = args.input if isinstance(args.input, str) else args.input[0]
    logger.info("prompt: %s" % text_prompt)

    logger.info('Start inference...')

    audio_array = generate_audio(models, text_prompt)

    logger.info('Script finished successfully.')


def main():
    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    models = {
        "net": net,
        "tokenizer": tokenizer,
    }

    # generate
    recognize_from_text(models)


if __name__ == '__main__':
    main()
