import argparse
import logging
import os
import re
import sys
import unicodedata

import ailia
from transformers import T5Tokenizer
from onnxt5 import GenerativeT5
sys.path.append('../../util')

# logger
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="T5 base Japanese title generation")
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
"""
params
"""
MODEL_NAME = "sonoisa/t5-base-japanese-title-generation"
ONNX_DIR = "/Users/t-ibayashi/Workspace/axinc/workspace/t5-base-japanese-title-generation/onnx"
MAX_SOURCE_LENGTH = 512

"""
pre process functions
"""
def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def normalize_text(text: str) -> str:
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def preprocess_body(text: str) -> str:
    return normalize_text(text.replace("\n", " "))

def main():
    body = """
    これはLEGOとRaspberry Piで実用的なスマートロックを作り上げる物語です。
    スマートロック・システムの全体構成は下図のようになります。図中右上にある塊が、全部LEGOで作られたスマートロックです。

    特徴は、3Dプリンタ不要で、LEGOという比較的誰でも扱えるもので作られたハードウェアであるところ、見た目の野暮ったさと機能のスマートさを兼ね備え、エンジニア心をくすぐるポイント満載なところです。
    なお、LEGO (レゴ)、LEGO Boost (ブースト) は LEGO Group (レゴグループ) の登録商標であり、この文書はレゴグループやその日本法人と一切関係はありません。

    次のようなシチュエーションを経験したことはありませんか？

    - 外出先にて、「そういや、鍵、閉めてきたかな？記憶がない…（ソワソワ）」
    - 朝の通勤にて、駅に到着してみたら「あ、鍵閉め忘れた。戻るか…」
    - 料理中に「あ、鍵閉め忘れた！でも、いま手が離せない。」
    - 玄関先で「手は買い物で一杯。ポケットから鍵を出すのが大変。」
    - 職場にて、夕方「そろそろ子供は家に帰ってきたかな？」
    - 玄関にて「今日は傘いるかな？」

    今回作るスマートロックは、次の機能でこれらを解決に導きます。

    - 鍵の閉め忘れをSlackに通知してくれる。iPhoneで施錠状態を確認できる。
    - 何処ででもiPhoneから施錠できる。
    - 「Hey Siri 鍵閉めて（鍵開けて）」で施錠/開錠できる。
    - 鍵の開閉イベントがiPhoneに通知され、帰宅が分かる。
    - LEDの色で天気予報（傘の必要性）を教えてくれる（ただし、時間の都合で今回は説明省略）。

    欲しくなりましたでしょうか？

    以下、ムービー多めで機能の詳細と作り方について解説していきます。ハードウェアもソフトウェアもオープンソースとして公開します。
    """
    args = parser.parse_args()

    # load model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, is_fast=True)

    if args.onnx:
        from onnxruntime import InferenceSession
        encoder_sess = InferenceSession(os.path.join(ONNX_DIR, "t5-base-japanese-title-generation-encoder.onnx"))
        decoder_sess = InferenceSession(os.path.join(ONNX_DIR, "t5-base-japanese-title-generation-decoder-with-lm-head.onnx"))
        model = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
    else:
        raise Exception("Only onnx runtime is supported")

    # pre process
    body_preprocessed = preprocess_body(body)
    most_plausible_title, _ = model(body_preprocessed, 21, temperature=0.)
    print(most_plausible_title)


if __name__ == '__main__':
    main()
