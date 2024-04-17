import re
import unicodedata
import sys
sys.path.append('..')

import pyopenjtalk
from num2words import num2words


# code is from text/japanese_mora_list.py, in vits2 japanese specialized version of the Bert-VITS2
"""
VOICEVOXのソースコードからお借りして最低限に改造したコード。
https://github.com/VOICEVOX/voicevox_engine/blob/master/voicevox_engine/tts_pipeline/mora_list.py
"""
"""
以下のモーラ対応表はOpenJTalkのソースコードから取得し、
カタカナ表記とモーラが一対一対応するように改造した。
ライセンス表記：
-----------------------------------------------------------------
          The Japanese TTS System "Open JTalk"
          developed by HTS Working Group
          http://open-jtalk.sourceforge.net/
-----------------------------------------------------------------

 Copyright (c) 2008-2014  Nagoya Institute of Technology
                          Department of Computer Science

All rights reserved.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.
- Neither the name of the HTS working group nor the names of its
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from typing import Optional

# (カタカナ, 子音, 母音)の順。子音がない場合はNoneを入れる。
# 但し「ン」と「ッ」は母音のみという扱いで、それぞれ「N」「q (clから変更)」
# また「デェ = dy e」はpyopenjtalkの出力（de e）と合わないため削除
_mora_list_minimum: list[tuple[str, Optional[str], str]] = [
    ("ヴォ", "v", "o"),
    ("ヴェ", "v", "e"),
    ("ヴィ", "v", "i"),
    ("ヴァ", "v", "a"),
    ("ヴ", "v", "u"),
    ("ン", None, "N"),
    ("ワ", "w", "a"),
    ("ロ", "r", "o"),
    ("レ", "r", "e"),
    ("ル", "r", "u"),
    ("リョ", "ry", "o"),
    ("リュ", "ry", "u"),
    ("リャ", "ry", "a"),
    ("リェ", "ry", "e"),
    ("リ", "r", "i"),
    ("ラ", "r", "a"),
    ("ヨ", "y", "o"),
    ("ユ", "y", "u"),
    ("ヤ", "y", "a"),
    ("モ", "m", "o"),
    ("メ", "m", "e"),
    ("ム", "m", "u"),
    ("ミョ", "my", "o"),
    ("ミュ", "my", "u"),
    ("ミャ", "my", "a"),
    ("ミェ", "my", "e"),
    ("ミ", "m", "i"),
    ("マ", "m", "a"),
    ("ポ", "p", "o"),
    ("ボ", "b", "o"),
    ("ホ", "h", "o"),
    ("ペ", "p", "e"),
    ("ベ", "b", "e"),
    ("ヘ", "h", "e"),
    ("プ", "p", "u"),
    ("ブ", "b", "u"),
    ("フォ", "f", "o"),
    ("フェ", "f", "e"),
    ("フィ", "f", "i"),
    ("ファ", "f", "a"),
    ("フ", "f", "u"),
    ("ピョ", "py", "o"),
    ("ピュ", "py", "u"),
    ("ピャ", "py", "a"),
    ("ピェ", "py", "e"),
    ("ピ", "p", "i"),
    ("ビョ", "by", "o"),
    ("ビュ", "by", "u"),
    ("ビャ", "by", "a"),
    ("ビェ", "by", "e"),
    ("ビ", "b", "i"),
    ("ヒョ", "hy", "o"),
    ("ヒュ", "hy", "u"),
    ("ヒャ", "hy", "a"),
    ("ヒェ", "hy", "e"),
    ("ヒ", "h", "i"),
    ("パ", "p", "a"),
    ("バ", "b", "a"),
    ("ハ", "h", "a"),
    ("ノ", "n", "o"),
    ("ネ", "n", "e"),
    ("ヌ", "n", "u"),
    ("ニョ", "ny", "o"),
    ("ニュ", "ny", "u"),
    ("ニャ", "ny", "a"),
    ("ニェ", "ny", "e"),
    ("ニ", "n", "i"),
    ("ナ", "n", "a"),
    ("ドゥ", "d", "u"),
    ("ド", "d", "o"),
    ("トゥ", "t", "u"),
    ("ト", "t", "o"),
    ("デョ", "dy", "o"),
    ("デュ", "dy", "u"),
    ("デャ", "dy", "a"),
    # ("デェ", "dy", "e"),
    ("ディ", "d", "i"),
    ("デ", "d", "e"),
    ("テョ", "ty", "o"),
    ("テュ", "ty", "u"),
    ("テャ", "ty", "a"),
    ("ティ", "t", "i"),
    ("テ", "t", "e"),
    ("ツォ", "ts", "o"),
    ("ツェ", "ts", "e"),
    ("ツィ", "ts", "i"),
    ("ツァ", "ts", "a"),
    ("ツ", "ts", "u"),
    ("ッ", None, "q"),  # 「cl」から「q」に変更
    ("チョ", "ch", "o"),
    ("チュ", "ch", "u"),
    ("チャ", "ch", "a"),
    ("チェ", "ch", "e"),
    ("チ", "ch", "i"),
    ("ダ", "d", "a"),
    ("タ", "t", "a"),
    ("ゾ", "z", "o"),
    ("ソ", "s", "o"),
    ("ゼ", "z", "e"),
    ("セ", "s", "e"),
    ("ズィ", "z", "i"),
    ("ズ", "z", "u"),
    ("スィ", "s", "i"),
    ("ス", "s", "u"),
    ("ジョ", "j", "o"),
    ("ジュ", "j", "u"),
    ("ジャ", "j", "a"),
    ("ジェ", "j", "e"),
    ("ジ", "j", "i"),
    ("ショ", "sh", "o"),
    ("シュ", "sh", "u"),
    ("シャ", "sh", "a"),
    ("シェ", "sh", "e"),
    ("シ", "sh", "i"),
    ("ザ", "z", "a"),
    ("サ", "s", "a"),
    ("ゴ", "g", "o"),
    ("コ", "k", "o"),
    ("ゲ", "g", "e"),
    ("ケ", "k", "e"),
    ("グヮ", "gw", "a"),
    ("グ", "g", "u"),
    ("クヮ", "kw", "a"),
    ("ク", "k", "u"),
    ("ギョ", "gy", "o"),
    ("ギュ", "gy", "u"),
    ("ギャ", "gy", "a"),
    ("ギェ", "gy", "e"),
    ("ギ", "g", "i"),
    ("キョ", "ky", "o"),
    ("キュ", "ky", "u"),
    ("キャ", "ky", "a"),
    ("キェ", "ky", "e"),
    ("キ", "k", "i"),
    ("ガ", "g", "a"),
    ("カ", "k", "a"),
    ("オ", None, "o"),
    ("エ", None, "e"),
    ("ウォ", "w", "o"),
    ("ウェ", "w", "e"),
    ("ウィ", "w", "i"),
    ("ウ", None, "u"),
    ("イェ", "y", "e"),
    ("イ", None, "i"),
    ("ア", None, "a"),
]
_mora_list_additional: list[tuple[str, Optional[str], str]] = [
    ("ヴョ", "by", "o"),
    ("ヴュ", "by", "u"),
    ("ヴャ", "by", "a"),
    ("ヲ", None, "o"),
    ("ヱ", None, "e"),
    ("ヰ", None, "i"),
    ("ヮ", "w", "a"),
    ("ョ", "y", "o"),
    ("ュ", "y", "u"),
    ("ヅ", "z", "u"),
    ("ヂ", "j", "i"),
    ("ヶ", "k", "e"),
    ("ャ", "y", "a"),
    ("ォ", None, "o"),
    ("ェ", None, "e"),
    ("ゥ", None, "u"),
    ("ィ", None, "i"),
    ("ァ", None, "a"),
]

# 例: "vo" -> "ヴォ", "a" -> "ア"
mora_phonemes_to_mora_kata: dict[str, str] = {
    (consonant or "") + vowel: kana for [kana, consonant, vowel] in _mora_list_minimum
}

# 例: "ヴォ" -> ("v", "o"), "ア" -> (None, "a")
mora_kata_to_mora_phonemes: dict[str, tuple[Optional[str], str]] = {
    kana: (consonant, vowel)
    for [kana, consonant, vowel] in _mora_list_minimum + _mora_list_additional
}

# code is from text/japanese.py, in vits2 japanese specialized version of the Bert-VITS2

punctuation = ["!", "?", "…", ",", ".", "'", "-"]

# 子音の集合
COSONANTS = set(
    [
        cosonant
        for cosonant, _ in mora_kata_to_mora_phonemes.values()
        if cosonant is not None
    ]
)

# 母音の集合
VOWELS = {"a", "i", "u", "e", "o"}


# 正規化で記号を変換するための辞書
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # "~": "-",  # これは長音記号「ー」として扱うよう変更
    "「": "'",
    "」": "'",
}


def text_normalize(text):
    """
    日本語のテキストを正規化する。
    結果は、ちょうど次の文字のみからなる：
    - ひらがな
    - カタカナ（全角長音記号「ー」が入る！）
    - 漢字
    - 半角アルファベット（大文字と小文字）
    - ギリシャ文字
    - `.` （句点`。`や`…`の一部や改行等）
    - `,` （読点`、`や`:`等）
    - `?` （疑問符`？`）
    - `!` （感嘆符`！`）
    - `'` （`「`や`」`等）
    - `-` （`―`（ダッシュ、長音記号ではない）や`-`等）

    注意点:
    - 三点リーダー`…`は`...`に変換される（`なるほど…。` → `なるほど....`）
    - 数字は漢字に変換される（`1,100円` → `千百円`、`52.34` → `五十二点三四`）
    - 読点や疑問符等の位置・個数等は保持される（`??あ、、！！！` → `??あ,,!!!`）
    """
    #print(f"Before normalization: {text}")
    # ここでアルファベットは半角になり、三点リーダは`...`になる
    res = unicodedata.normalize("NFKC", text)

    res = japanese_convert_numbers_to_words(res)  # 「100円」→「百円」等

    # 「～」と「~」も長音記号として扱う
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")

    res = replace_punctuation(res)  # 句読点等正規化、読めない文字を削除

    # 結合文字の濁点・半濁点を削除
    # 通常の「ば」等はそのままのこされる、「あ゛」は上で「あ゙」になりここで「あ」になる
    res = res.replace("\u3099", "")  # 結合文字の濁点を削除、る゙ → る
    res = res.replace("\u309A", "")  # 結合文字の半濁点を削除、な゚ → な
    return res


def replace_punctuation(text: str) -> str:
    """句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalkで読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、アルファベット、ギリシャ文字
    """
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    # 句読点を辞書で置換
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        # ↓ ひらがな、カタカナ、漢字
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        # ↓ 半角アルファベット（大文字と小文字）
        + r"\u0041-\u005A\u0061-\u007A"
        # ↓ 全角アルファベット（大文字と小文字）
        + r"\uFF21-\uFF3A\uFF41-\uFF5A"
        # ↓ ギリシャ文字
        + r"\u0370-\u03FF\u1F00-\u1FFF"
        # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
        + "".join(punctuation) + r"]+",
        # 上述以外の文字を削除
        "",
        replaced_text,
    )

    return replaced_text


_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")


def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def g2p(norm_text: str, tokenizer) -> tuple[list[str], list[int], list[int]]:
    """
    他で使われるメインの関数。`text_normalize()`で正規化された`norm_text`を受け取り、
    - phones: 音素のリスト（ただし`!`や`,`や`.`等punctuationが含まれうる）
    - tones: アクセントのリスト、0（低）と1（高）からなり、phonesと同じ長さ
    - word2ph: 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
    のタプルを返す。
    ただし`phones`と`tones`の最初と終わりに`_`が入り、応じて`word2ph`の最初と最後に1が追加される。
    """
    # pyopenjtalkのフルコンテキストラベルを使ってアクセントを取り出すと、punctuationの位置が消えてしまい情報が失われてしまう：
    # 「こんにちは、世界。」と「こんにちは！世界。」と「こんにちは！！！？？？世界……。」は全て同じになる。
    # よって、まずpunctuation無しの音素とアクセントのリストを作り、
    # それとは別にpyopenjtalk.run_frontend()で得られる音素リスト（こちらはpunctuationが保持される）を使い、
    # アクセント割当をしなおすことによってpunctuationを含めた音素とアクセントのリストを作る。

    # punctuationがすべて消えた、音素とアクセントのタプルのリスト
    phone_tone_list_wo_punct = g2phone_tone_wo_punct(norm_text)

    # sep_text: 単語単位の単語のリスト
    # sep_kata: 単語単位の単語のカタカナ読みのリスト
    sep_text, sep_kata = text2sep_kata(norm_text)

    # sep_phonemes: 各単語ごとの音素のリストのリスト
    sep_phonemes = handle_long([kata2phoneme_list(i) for i in sep_kata])

    # phone_w_punct: sep_phonemesを結合した、punctuationを元のまま保持した音素列
    phone_w_punct: list[str] = []
    for i in sep_phonemes:
        phone_w_punct += i

    # punctuation無しのアクセント情報を使って、punctuationを含めたアクセント情報を作る
    phone_tone_list = align_tones(phone_w_punct, phone_tone_list_wo_punct)
    # word2phは厳密な解答は不可能なので（「今日」「眼鏡」等の熟字訓が存在）、
    # Bert-VITS2では、単語単位の分割を使って、単語の文字ごとにだいたい均等に音素を分配する

    # sep_textから、各単語を1文字1文字分割して、文字のリスト（のリスト）を作る
    sep_tokenized: list[list[str]] = []
    for i in sep_text:
        if i not in punctuation:
            sep_tokenized.append(tokenizer.tokenize(i))  # ここでおそらく`i`が文字単位に分割される
        else:
            sep_tokenized.append([i])

    # 各単語について、音素の数と文字の数を比較して、均等っぽく分配する
    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        word2ph += distribute_phone(phone_len, word_len)

    # 最初と最後に`_`記号を追加、アクセントは0（低）、word2phもそれに合わせて追加
    phone_tone_list = [("_", 0)] + phone_tone_list + [("_", 0)]
    word2ph = [1] + word2ph + [1]

    phones = [phone for phone, _ in phone_tone_list]
    tones = [tone for _, tone in phone_tone_list]

    assert len(phones) == sum(word2ph), f"{len(phones)} != {sum(word2ph)}"

    return phones, tones, word2ph


def g2phone_tone_wo_punct(text: str) -> list[tuple[str, int]]:
    """
    テキストに対して、音素とアクセント（0か1）のペアのリストを返す。
    ただし「!」「.」「?」等の非音素記号(punctuation)は全て消える（ポーズ記号も残さない）。
    非音素記号を含める処理は`align_tones()`で行われる。
    また「っ」は「cl」でなく「q」に変換される（「ん」は「N」のまま）。
    例: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]
    """
    prosodies = pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0
    for i, letter in enumerate(prosodies):
        # 特殊記号の処理

        # 文頭記号、無視する
        if letter == "^":
            assert i == 0, "Unexpected ^"
        # アクセント句の終わりに来る記号
        elif letter in ("$", "?", "_", "#"):
            # 保持しているフレーズを、アクセント数値を0-1に修正し結果に追加
            result.extend(fix_phone_tone(current_phrase))
            # 末尾に来る終了記号、無視（文中の疑問文は`_`になる）
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            # あとは"_"（ポーズ）と"#"（アクセント句の境界）のみ
            # これらは残さず、次のアクセント句に備える。
            current_phrase = []
            # 0を基準点にしてそこから上昇・下降する（負の場合は上の`fix_phone_tone`で直る）
            current_tone = 0
        # アクセント上昇記号
        elif letter == "[":
            current_tone = current_tone + 1
        # アクセント下降記号
        elif letter == "]":
            current_tone = current_tone - 1
        # それ以外は通常の音素
        else:
            if letter == "cl":  # 「っ」の処理
                letter = "q"
            current_phrase.append((letter, current_tone))
    return result


def text2sep_kata(norm_text: str) -> tuple[list[str], list[str]]:
    """
    `text_normalize`で正規化済みの`norm_text`を受け取り、それを単語分割し、
    分割された単語リストとその読み（カタカナor記号1文字）のリストのタプルを返す。
    単語分割結果は、`g2p()`の`word2ph`で1文字あたりに割り振る音素記号の数を決めるために使う。
    例:
    `私はそう思う!って感じ?` →
    ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]
    """
    # parsed: OpenJTalkの解析結果
    parsed = pyopenjtalk.run_frontend(norm_text)
    sep_text: list[str] = []
    sep_kata: list[str] = []
    for parts in parsed:
        # word: 実際の単語の文字列
        # yomi: その読み、但し無声化サインの`’`は除去
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        """
        ここで`yomi`の取りうる値は以下の通りのはず。
        - `word`が通常単語 → 通常の読み（カタカナ）
            （カタカナからなり、長音記号も含みうる、`アー` 等）
        - `word`が`ー` から始まる → `ーラー` や `ーーー` など
        - `word`が句読点や空白等 → `、`
        - `word`が`?` → `？`（全角になる）
        他にも`word`が読めないキリル文字アラビア文字等が来ると`、`になるが、正規化でこの場合は起きないはず。
        また元のコードでは`yomi`が空白の場合の処理があったが、これは起きないはず。
        処理すべきは`yomi`が`、`の場合のみのはず。
        """
        assert yomi != "", f"Empty yomi: {word}"
        if yomi == "、":
            # wordは正規化されているので、`.`, `,`, `!`, `'`, `-`のいずれか
            if word not in (
                ".",
                ",",
                "!",
                "'",
                "-",
            ):
                # ここはpyopenjtalkが読めない文字等のときに起こる
                raise ValueError(f"Cannot read: {word} in:\n{norm_text}")
            # yomiは元の記号のままに変更
            yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)
    return sep_text, sep_kata


# ESPnetの実装から引用、変更点無し
# https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> list[str]:
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list`のtone（アクセントの値）を0か1の範囲に修正する。
    例: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]
    """
    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """
    左から右に1ずつ振り分け、次にまた左から右に1ずつ増やし、というふうに、
    音素の数`n_phone`を単語の数`n_word`に分配する。
    """
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    return sep_phonemes


def align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """
    例:
    …私は、、そう思う。
    phones_with_punct:
    [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
    [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
    [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]
    """
    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            # 余ったpunctuationがある場合 → (punctuation, 0)を追加
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            # phone_tone_listの現在の音素と一致する場合 → toneをそこから取得、(phone, tone)を追加
            result.append((phone, phone_tone_list[tone_index][1]))
            # 探すindexを1つ進める
            tone_index += 1
        elif phone in punctuation:
            # phoneがpunctuationの場合 → (phone, 0)を追加
            result.append((phone, 0))
        else:
            print(f"phones: {phones_with_punct}")
            print(f"phone_tone_list: {phone_tone_list}")
            print(f"result: {result}")
            print(f"tone_index: {tone_index}")
            print(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")
    return result


def kata2phoneme_list(text: str) -> list[str]:
    """
    原則カタカナの`text`を受け取り、それをそのままいじらずに音素記号のリストに変換。
    注意点：
    - punctuationが来た場合（punctuationが1文字の場合がありうる）、処理せず1文字のリストを返す
    - 冒頭に続く「ー」はそのまま「ー」のままにする（`handle_long()`で処理される）
    - 文中の「ー」は前の音素記号の最後の音素記号に変換される。
    例：
    `ーーソーナノカーー` → ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    `?` → ["?"]
    """
    if text in punctuation:
        return [text]
    # `text`がカタカナ（`ー`含む）のみからなるかどうかをチェック
    if re.fullmatch(r"[\u30A0-\u30FF]+", text) is None:
        raise ValueError(f"Input must be katakana only: {text}")
    sorted_keys = sorted(mora_kata_to_mora_phonemes.keys(), key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_keys))

    def mora2phonemes(mora: str) -> str:
        cosonant, vowel = mora_kata_to_mora_phonemes[mora]
        if cosonant is None:
            return f" {vowel}"
        return f" {cosonant} {vowel}"

    spaced_phonemes = re.sub(pattern, lambda m: mora2phonemes(m.group()), text)

    # 長音記号「ー」の処理
    long_pattern = r"(\w)(ー*)"
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))
    spaced_phonemes = re.sub(long_pattern, long_replacement, spaced_phonemes)
    return spaced_phonemes.strip().split(" ")



def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

from symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids