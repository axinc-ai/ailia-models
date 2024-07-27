# WhisperTokenizer._decode_asr

import json
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import regex as re

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

TASK_IDS = ["translate", "transcribe"]

def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    # It would be much harder to do O(n) because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    # If token_timestamp_sequences is provided, will split those sequences in
    # exactly the same way.

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    if token_timestamp_sequences:
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        total_token_timestamp_sequence = []

    for seq_idx, right_sequence in enumerate(sequences[1:]):
        # index = 0
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        # Here we're sliding matches
        # [a, b, c, d]
        #          [c, d, f]
        # =        [c] == [d]
        #
        # [a, b, c, d]
        #       [c, d, f]
        # =     [c, d] == [c, d]
        #
        #
        # [a, b, c, d]
        #    [c, d, f]
        #
        # =  [b, c, d] == [c, d, f]
        #
        # [a, b, c, d]
        # [c, d, f]
        #
        # [a, b, c] == [c, d, f]
        #
        # [a, b, c, d]
        # [d, f]
        #
        # [a, b] == [d, f]
        #
        # [a, b, c, d]
        # [f]
        #
        # [a] == [f]
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            # epsilon to favor long perfect matches
            eps = i / 10000.0

            # Slightly convoluted because we don't want out of bound indices
            # This will be necessary for a small conflict resolution optimization
            # later
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])

            # We can only match subsequences of the same size.
            if len(left) != len(right):
                raise RuntimeError(
                    "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                )

            matches = np.sum(left == right)
            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        (left_start, left_stop, right_start, right_stop) = max_indices

        # This is a small conflict optimization since those sequences overlap
        # in audio.
        # We're going to give more confidence to the left sequence
        # for the left of the overlap,
        # and to the right of the sequence, for the right of the overlap
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]

    total_sequence.extend(left_sequence)

    if token_timestamp_sequences is None:
        return total_sequence

    if len(token_timestamp_sequences) > 0:
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        return total_sequence, total_token_timestamp_sequence
    else:
        return total_sequence, []


def _collate_word_timestamps(tokenizer, tokens, token_timestamps, language, return_language):
    words, _, token_indices = _combine_tokens_into_words(tokenizer, tokens, language)

    optional_language_field = {"language": language} if return_language else {}

    timings = [
        {
            "text": word,
            "timestamp": (token_timestamps[indices[0]][0], token_timestamps[indices[-1]][1]),
            **optional_language_field,
        }
        for word, indices in zip(words, token_indices)
    ]
    return timings


def _combine_tokens_into_words(
    tokenizer,
    tokens: List[int],
    language: str = None,
    prepend_punctuations: str = "\"'“¡¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
):
    """
    Groups tokens by word. Returns a tuple containing a list of strings with the words, and a list of `token_id`
    sequences with the tokens making up each word.
    """
    if language is None:
        language = tokenizer.language
    if language is None:
        language = "english"

    if language in {"chinese", "japanese", "thai", "lao", "myanmar", "cantonese"}:
        # These languages don't typically use spaces.
        words, word_tokens, token_indices = _split_tokens_on_unicode(tokenizer, tokens)
    else:
        words, word_tokens, token_indices = _split_tokens_on_spaces(tokenizer, tokens)

    _merge_punctuations(words, word_tokens, token_indices, prepend_punctuations, append_punctuations)
    return words, word_tokens, token_indices


def _split_tokens_on_unicode(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points."""
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "\ufffd"

    words = []
    word_tokens = []
    token_indices = []
    current_tokens = []
    current_indices = []
    unicode_offset = 0

    for token_idx, token in enumerate(tokens):
        current_tokens.append(token)
        current_indices.append(token_idx)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char
        ):
            words.append(decoded)
            word_tokens.append(current_tokens)
            token_indices.append(current_indices)
            current_tokens = []
            current_indices = []
            unicode_offset += len(decoded)

    return words, word_tokens, token_indices


def _split_tokens_on_spaces(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at whitespace and punctuation tokens."""
    subwords, subword_tokens_list, subword_indices_list = _split_tokens_on_unicode(tokenizer, tokens)
    words = []
    word_tokens = []
    token_indices = []

    for subword, subword_tokens, subword_indices in zip(subwords, subword_tokens_list, subword_indices_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
            token_indices.append(subword_indices)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
            token_indices[-1].extend(subword_indices)

    return words, word_tokens, token_indices


def _merge_punctuations(words, tokens, indices, prepended, appended):
    """Merges punctuation tokens with neighboring words."""
    # prepend punctuations
    i = len(words) - 2
    j = len(words) - 1
    while i >= 0:
        if words[i].startswith(" ") and words[i].strip() in prepended:
            words[j] = words[i] + words[j]
            tokens[j] = tokens[i] + tokens[j]
            indices[j] = indices[i] + indices[j]
            words[i] = ""
            tokens[i] = []
            indices[i] = []
        else:
            j = i
        i -= 1

    # append punctuations
    i = 0
    j = 1
    while j < len(words):
        if not words[i].endswith(" ") and words[j] in appended:
            words[i] += words[j]
            tokens[i] += tokens[j]
            indices[i] += indices[j]
            words[j] = ""
            tokens[j] = []
            indices[j] = []
        else:
            i = j
        j += 1

    # remove elements that are now empty
    words[:] = [word for word in words if word]
    tokens[:] = [token for token in tokens if token]
    indices[:] = [idx for idx in indices if idx]

def decode_asr(tokenizer, model_outputs, *, return_timestamps, return_language, time_precision):
    """
    Internal method meant to only be used by asr pipeline. Handles all the little quirks specific to whisper to handle
    the various options not allowed in other seq2seq models
    """

    # =========== Overview ============
    # - iterate over all outputs
    # - all tokens within output
    # - Each token can be
    #   - language token
    #   - special token
    #   - timestamp token
    #   - text token
    # - We accumulate the text tokens.
    # - We split on end timestamps
    # - Lots of complexity comes from stride and timestamps

    last_language = None

    def new_chunk():
        return {"language": last_language, "timestamp": [None, None], "text": ""}

    # Welcome to the state machine !
    chunks = []
    chunk = new_chunk()
    time_offset = 0.0
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    previous_tokens = []
    previous_token_timestamps = []
    skip = False
    right_stride_start = None

    all_special_ids = set(tokenizer.all_special_ids)
    prompt_token_id = tokenizer.convert_tokens_to_ids("<|startofprev|>")
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    # - iterate over all outputs
    for chunk_id, output in enumerate(model_outputs):
        # We can drop everything to Python list, it's going to make
        # our lives easier
        token_ids = output["tokens"][0].tolist()
        # (possibly) remove the prompt from the token ids
        token_ids = tokenizer._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)
        if return_timestamps == "word":
            token_timestamps = output["token_timestamps"][0].tolist()

        # Those keep track of timestamps within strides
        # Which need to be skipped and resolve all tokens in a single
        # chunk.
        last_timestamp = None
        first_timestamp = timestamp_begin

        if "stride" in output:
            chunk_len, stride_left, stride_right = output["stride"]
            # Offset the timings to account for the other `model_outputs`.
            time_offset -= stride_left
            right_stride_start = chunk_len - stride_right

            # Keeping track of timestamps within strides
            # We're going to NOT split on those, and delay until we're
            # out of BOTH stride. Otherwise lots of issues occur and
            # corner cases
            if stride_left:
                first_timestamp = stride_left / time_precision + timestamp_begin
            if stride_right:
                for token in reversed(token_ids):
                    if token >= timestamp_begin:
                        # There can be several token in the right stride
                        # But the last one is ALWAYS going to be skipped
                        if (
                            last_timestamp is not None
                            and (token - timestamp_begin) * time_precision < right_stride_start
                        ):
                            break
                        last_timestamp = token

        current_tokens = []
        current_token_timestamps = []

        # - all tokens within output
        for i, token in enumerate(token_ids):
            # 4 possible states for each token
            # - 1/ Language code
            # - 2/ all other special tokens (which we ignore)
            # - 3/ Timestamp
            # - 4/ Regular text
            if token in all_special_ids:
                # Either language code or other
                text = tokenizer.decode([token])
                # Removing outer shell <|XX|>
                text = text[2:-2]
                language = LANGUAGES.get(text, None)
                if language is not None:
                    # 1/ Indeed some language
                    # TODO Handle when language is different from the previous
                    # one, and we cannot use timestamped tokens to create chunks
                    if last_language and language != last_language and not return_timestamps:
                        previous_tokens.append(current_tokens)
                        resolved_tokens = _find_longest_common_sequence(previous_tokens)
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        chunk = new_chunk()
                    chunk["language"] = language
                    last_language = language
                else:
                    # 2/ This is a regular special token, ignoring it
                    pass
            elif token >= timestamp_begin:
                # 3/ Timestamp token
                time = (token - timestamp_begin) * time_precision + time_offset
                time = round(time, 2)
                if last_timestamp and token >= last_timestamp:
                    # Whisper outputted a timestamp token, but it falls within
                    # our stride, so we're going to skip it for the time being
                    # and resolve this later
                    # Skip is necessary because timestamp tokens always come
                    # by pair, so we need to skip the next one too (which would mark the start of another chunk).
                    skip = True
                elif skip or (previous_tokens and token < first_timestamp):
                    skip = False
                elif chunk["timestamp"][0] is None:
                    chunk["timestamp"][0] = time
                else:
                    # This is the end of the timestamp chunk
                    if time == chunk["timestamp"][0]:
                        # This is a bug in timestamp token output
                        # where we're taking the duplicate token
                        # as a stop where it should be a start.
                        # This is an issue in the underlying model output
                        # Let's just skip it so it becomes de-factor
                        # a start agin
                        pass
                    else:
                        chunk["timestamp"][1] = time
                        # Handling merges.
                        previous_tokens.append(current_tokens)
                        if return_timestamps == "word":
                            previous_token_timestamps.append(current_token_timestamps)
                        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
                            previous_tokens, previous_token_timestamps
                        )
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        if return_timestamps == "word":
                            chunk["words"] = _collate_word_timestamps(
                                tokenizer, resolved_tokens, resolved_token_timestamps, last_language, return_language
                            )
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        previous_token_timestamps = []
                        current_token_timestamps = []
                        chunk = new_chunk()
            else:
                # 4/ Regular token
                # We just append to the list of all tokens so we can handle
                # merges later and decode into text.
                current_tokens.append(token)
                if return_timestamps == "word":
                    start_time = round(token_timestamps[i] + time_offset, 2)
                    if i + 1 < len(token_timestamps):
                        end_time = round(token_timestamps[i + 1] + time_offset, 2)
                    else:
                        end_time = None  # should never happen
                    current_token_timestamps.append((start_time, end_time))

        if "stride" in output:
            time_offset += chunk_len - stride_right

        # Leftover tokens
        if current_tokens:
            previous_tokens.append(current_tokens)
            if return_timestamps == "word":
                previous_token_timestamps.append(current_token_timestamps)
        elif not (any(p for p in previous_tokens)):
            chunk = new_chunk()
            previous_tokens = []
            current_tokens = []
            previous_token_timestamps = []
            current_token_timestamps = []

    if previous_tokens:
        if return_timestamps:
            logger.warning(
                "Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. "
                "Also make sure WhisperTimeStampLogitsProcessor was used during generation."
            )
        # Happens when we don't use timestamps
        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
            previous_tokens, previous_token_timestamps
        )
        resolved_text = tokenizer.decode(resolved_tokens)
        chunk["text"] = resolved_text
        if return_timestamps == "word":
            chunk["words"] = _collate_word_timestamps(
                tokenizer, resolved_tokens, resolved_token_timestamps, last_language, return_language
            )
        chunks.append(chunk)

    # Preparing and cleaning up the pipeline output
    full_text = "".join(chunk["text"] for chunk in chunks)
    if return_timestamps or return_language:
        for chunk in chunks:
            if not return_timestamps:
                chunk.pop("timestamp")
            else:
                chunk["timestamp"] = tuple(chunk["timestamp"])
            if not return_language:
                chunk.pop("language")

        if return_timestamps == "word":
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunk["words"])
            optional = {"chunks": new_chunks}
        else:
            optional = {"chunks": chunks}
    else:
        optional = {}
    return full_text, optional