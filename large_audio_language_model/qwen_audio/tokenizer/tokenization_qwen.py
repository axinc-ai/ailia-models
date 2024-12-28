# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenization classes for QWen."""

import base64
import logging
import os
import re
import itertools

import requests
import unicodedata
from typing import Collection, Dict, List, Set, Tuple, Union, Any, Callable, Optional

import tiktoken
import numpy as np

from transformers import PreTrainedTokenizer, AddedToken
from transformers.utils import try_to_load_from_cache
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy, \
    TextInput, TextInputPair, PreTokenizedInput, PreTokenizedInputPair, TensorType, EncodedInput, EncodedInputPair

import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "qwen.tiktoken", "ttf": "SimSun.ttf"}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (
                     ENDOFTEXT,
                     IMSTART,
                     IMEND,
                 ) + EXTRAS

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "it": "italian",
}


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


def _list_find(
        input_list: List[Any],
        candidates: Tuple[Any],
        start: int = 0,
):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1


def _replace_closed_tag(
        input_tokens: List[Any],
        start_tags: Union[Any, Tuple[Any]],
        end_tags: Union[Any, Tuple[Any]],
        inclusive_replace_func: Callable,
        exclusive_replace_func: Callable = lambda x: x,
        audio_info: Dict = None
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)

    output_tokens = []
    end = 0
    audio_idx = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end: start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed audio token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start: end + 1], audio_info, audio_idx))
        end += 1
        audio_idx += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))
    return output_tokens


class QWenTokenizer(PreTrainedTokenizer):
    """QWen tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
            self,
            vocab_file,
            errors="replace",
            audio_start_tag='<audio>',
            audio_end_tag='</audio>',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_start_tag = audio_start_tag
        self.audio_end_tag = audio_end_tag
        self.audio_pad_tag = "[[[AUDIO:modality]]]"

        self.AUDIO_ST = (
            '[[[AUDIO:modality]]]',
            # Transcription Tag
            "<|startoftranscript|>",  # Transcription
            "<|startofanalysis|>",  # Analysis
            # Task Tag
            "<|translate|>",
            "<|transcribe|>",
            "<|caption|>",
            "<|keyword|>",
            # Language Tag
            "<|unknown|>",  # unknown language
            *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
            "<|zh_tr|>",  # tranditional Chinese
            # Timestamps Tag
            "<|notimestamps|>",
            "<|sil|>",
            "<|timestamps|>",
            *[f"<|{i * 0.01:.2f}|>" for i in range(3001)],  # timestamps 0.00-30.00
            # Output Instruction
            "<|caption_audiocaps|>",  # Audiocaps caption style
            "<|caption_clotho|>",  # Clotho caption style
            "<|audioset_ontology|>",  # Audioset ontology style
            "<|caption_plain|>",  # plain caption
            "<|itn|>",  # inversed text normalized
            "<|wo_itn|>",  # without inversed text normalized
            "<|startofentityvalue|>",
            "<|endofentityvalue|>",
            "<|startofentitytype|>",
            "<|endofentitytype|>",
            "<|named_entity_recognition|>",  # named entity recognition task
            "<|audio_grounding|>",
            "<|startofword|>",
            "<|endofword|>",
            "<|delim|>",  # delimiter of timestamps pair in audio grounding
            "<|emotion_recognition|>",  # emotion recognition
            "<|music_description|>",  # music description
            "<|note_analysis|>",  # note analysis
            "<|pitch|>",  # note analysis: pitch
            *[f"<|midi_pitch_{i}|>" for i in range(128)],  # midi pitch 0-127
            "<|velocity|>",  # note analysis: velocity
            *[f"<|midi_velocity_{i}|>" for i in range(128)],  # midi velocity 0-127
            "<|sonic|>",  # note analysis:  sonic
            "<|instrument|>",  # note analysis:  instrument
            "<|speaker_meta|>",  # meta information of speaker
            "<|song_meta|>",  # meta information of song
            "<|question|>",  # AQA: question
            "<|answer|>",  # AQA: answer
            "<|choice|>",  # AQA: answer choice
            "<|scene|>",  # scene recognition
            "<|event|>",  # sound event
            "<|vocal_classification|>",  # vocal classification
            "<|speech_understanding|>",  # speech language understanding
            "<|scenario|>",  # speech language understanding: scenario
            "<|action|>",  # speech language understanding: action
            "<|entities|>",  # speech language understanding: entities
            "<|speech_edit|>",  # speech edit
            audio_start_tag,
            audio_end_tag
        )

        self.errors = errors  # how to handle errors in decoding

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        self.special_tokens = {
            token: index
            for index, token in enumerate(
                SPECIAL_TOKENS + self.AUDIO_ST, start=len(self.mergeable_ranks)

            )
        }
        self.audio_start_id = self.special_tokens[self.audio_start_tag]
        self.audio_end_id = self.special_tokens[self.audio_end_tag]
        self.audio_pad_id = self.special_tokens[self.audio_pad_tag]
        print(f"audio_start_id: {self.audio_start_id}, "
              f"audio_end_id: {self.audio_end_id}, "
              f"audio_pad_id: {self.audio_pad_id}.")

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
                len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

    def __getstate__(self):
        # for pickle lovers
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return self.mergeable_ranks

    def convert_tokens_to_ids(
            self, tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        if not special_tokens and new_tokens:
            raise ValueError('Adding regular tokens is not supported')
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS  + self.AUDIO_ST:
                raise ValueError('Adding unknown special tokens is not supported')
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "qwen.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
            self,
            text: str,
            allowed_special: Union[Set, str] = "all",
            disallowed_special: Union[Collection, str] = (),
            audio_info: Dict = None,
            **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
                text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])

        def _encode_audiourl(audio_tokens, audio_info, audio_idx):
            assert audio_tokens[0] == self.audio_start_tag and audio_tokens[-1] == self.audio_end_tag
            audio_token_span = audio_info['audio_span_tokens'][audio_idx]
            out_audio_tokens = [self.audio_start_tag] + [self.audio_pad_tag] * (audio_token_span - 2) + [
                self.audio_end_tag]
            return out_audio_tokens

        return _replace_closed_tag(tokens, self.audio_start_tag, self.audio_end_tag, _encode_audiourl,
                                   audio_info=audio_info)

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        audio_info = kwargs.pop("audio_info", None)
        for pair_id in range(len(batch_text_or_text_pairs)):
            kwargs['audio_info'] = audio_info[pair_id]
            ids_or_pair_ids = batch_text_or_text_pairs[pair_id]
            # for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            errors: str = None,
            **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        audio_info = kwargs.pop("audio_info", None)

        def _decode_audiourl(audio_token_ids, audio_info, audio_idx):
            assert audio_token_ids[0] == self.audio_start_id and audio_token_ids[-1] == self.audio_end_id
            audio_url = audio_info["audio_urls"][audio_idx]
            return [self.audio_start_id] + self.tokenizer.encode(audio_url) + [self.audio_end_id]

        token_ids = _replace_closed_tag(token_ids, self.audio_start_id, self.audio_end_id, _decode_audiourl,
                                        audio_info=audio_info)

        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def to_list_format(self, text: str):
        text = unicodedata.normalize("NFC", text)
        token_ids = self.tokenizer.encode(
            text, allowed_special=set(self.AUDIO_ST + (ENDOFTEXT,)))

        def _encode_audio_info(tokens):
            if len(tokens) == 0:
                return []
            if tokens[0] == self.audio_start_id and tokens[-1] == self.audio_end_id:
                key = 'audio'
            else:
                _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
                return [{'text': b''.join(map(_tobytes, map(self.decoder.get, tokens))).decode('utf-8')}]
            _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
            val = b''.join(map(_tobytes, map(self.decoder.get, tokens[1:-1]))).decode('utf-8')
            return [{key: val}]

        return _replace_closed_tag(
            token_ids,
            (self.audio_start_id),
            (self.audio_end_id),
            _encode_audio_info,
            _encode_audio_info,
        )

    def from_list_format(self, list_format: List[Dict]):
        text = ''
        num_audios = 0
        for ele in list_format:
            if 'audio' in ele:
                num_audios += 1
                text += f'Audio {num_audios}:'
                text += self.audio_start_tag + ele['audio'] + self.audio_end_tag
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text

    def extract_audio_urls(self, text):
        pattern = rf"{self.audio_start_tag}(.*?){self.audio_end_tag}"
        return re.findall(pattern, text)

    def process_audio(self, text):
        audio_urls = self.extract_audio_urls(text)
        if len(audio_urls) > 0:
            audios, audio_lens, audio_span_tokens = [], [], []
            for audio_path in audio_urls:
                if audio_path.startswith("http://") or audio_path.startswith("https://"):  # http
                    data = bytes(requests.get(audio_path, stream=True).content)
                    audio = load_bytesio_audio(data)
                else:
                    audio = load_audio(audio_path)
                L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
                mel_len = L // 160
                audio = pad_or_trim(audio.flatten())
                mel = log_mel_spectrogram(audio)
                audio_len_after_cnn = get_T_after_cnn(mel_len)
                audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
                audio_len = [audio_len_after_cnn, audio_token_num]
                audios.append(mel)
                audio_lens.append(audio_len)
                audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos
            input_audio_lengths = torch.IntTensor(audio_lens)
            input_audios = torch.stack(audios, dim=0)
            return {"input_audios": input_audios,
                    "input_audio_lengths": input_audio_lengths,
                    "audio_span_tokens": audio_span_tokens,
                    "audio_urls": audio_urls}
        else:
            return None





