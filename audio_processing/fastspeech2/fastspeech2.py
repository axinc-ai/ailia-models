import ailia
import numpy as np
import yaml
import sys
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wavfile
from g2p_en import G2p
from pypinyin import pinyin, Style
from text import text_to_sequence
import re

# ===========================
# Settings
# ===========================

# リポジトリのルートにあるutilsを参照できるようにする
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

# モデル設定
WEIGHT_PATH_FS2 = './onnx/fastspeech2/ljspeech.onnx'
MODEL_PATH_FS2 = None
WEIGHT_PATH_HIFI = './onnx/hifigan/hifigan.onnx'
MODEL_PATH_HIFI = None
REMOTE_PATH = ""

PREPROCESS_CONFIG = "config/LJSpeech/preprocess.yaml"

# ★重要: エクスポート時と同じ最大長 (VRAM不足回避のため 600 で統一)
MODEL_MAX_LENGTH = 600

# 音声途切れを防ぐためのバッファ (10フレーム ≈ 0.1秒)
MEL_BUFFER_FRAMES = 40

# ===========================
# Arguments
# ===========================
parser = get_base_parser(
    'FastSpeech2',
    'fastspeech2.py',
    'output.wav'
)
# 元のFastSpeech2リポジトリと同じ引数名
parser.add_argument(
    '--source',
    type=str,
    default=None,
    help='path to a source file with format like train.txt and val.txt'
)
parser.add_argument(
    '--restore_step',
    type=int,
    required=False,
    default=900000,
    help='step for checkpoint to restore'
)
parser.add_argument(
    '--mode',
    type=str,
    choices=['batch', 'single'],
    required=False,
    default='single',
    help='Synthesize a whole dataset or a single sentence'
)
parser.add_argument(
    '-t', '--text',
    type=str,
    default="Ailia SDK makes it easy to deploy deep learning models.",
    help='raw text to synthesize, for single-sentence mode only'
)
parser.add_argument(
    '--speaker_id',
    type=int,
    default=0,
    help='speaker ID for multi-speaker synthesis, for single-sentence mode only'
)
parser.add_argument(
    '-p', '--pitch_control',
    type=float,
    default=1.0,
    help='control the pitch of the whole utterance, larger value for higher pitch'
)
parser.add_argument(
    '-e', '--energy_control',
    type=float,
    default=1.0,
    help='control the energy of the whole utterance, larger value for larger volume'
)
parser.add_argument(
    '-d', '--duration_control',
    type=float,
    default=1.0,
    help='control the speed of the whole utterance, larger value for slower speaking rate'
)
# ailia固有の引数
parser.add_argument(
    '--preprocess_config',
    type=str,
    default=PREPROCESS_CONFIG,
    help='path to preprocess.yaml'
)
parser.add_argument(
    '--model_config',
    type=str,
    default='config/LJSpeech/model.yaml',
    help='path to model.yaml'
)
parser.add_argument(
    '--onnx_fs2',
    default=WEIGHT_PATH_FS2,
    help='Path to FastSpeech2 ONNX file.'
)
parser.add_argument(
    '--onnx_hifi',
    default=WEIGHT_PATH_HIFI,
    help='Path to HiFi-GAN ONNX file.'
)
args = update_parser(parser)


# ===========================
# 2. 前処理(英語と中国語で異なる)
# ===========================
def preprocess_english(text, preprocess_config):
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w not in [" ", ""]:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    
    print(f"Phonemes: {phones}")
    
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence

def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print(f"Phonemes: {phones}")

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def get_preprocess_method(preprocess_config):
    dataset = preprocess_config["dataset"]
    if dataset == "LJSpeech":
        return preprocess_english
    if dataset == "LibriTTS":
        return preprocess_english
    if dataset == "AISHELL3":
        return preprocess_mandarin
    # デフォルトは英語とする
    return preprocess_english

# ===========================
# 3. Main Inference
# ===========================
def infer():
    # モデルのダウンロード
    check_and_download_models(args.onnx_fs2, MODEL_PATH_FS2, REMOTE_PATH)
    check_and_download_models(args.onnx_hifi, MODEL_PATH_HIFI, REMOTE_PATH)

    print("Loading Config...")
    # preprocess_configを読み込み
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    
    print("Loading ONNX Models...")
    env_id = args.env_id
    
    # ailia.Netの初期化
    fs2_net = ailia.Net(MODEL_PATH_FS2, args.onnx_fs2, env_id=env_id)
    hifi_net = ailia.Net(MODEL_PATH_HIFI, args.onnx_hifi, env_id=env_id)

    # -------------------------------------------
    # 入力データの準備（パディング処理）
    # -------------------------------------------
    # sourceファイルがあればそこから読み込み、なければtextを使用
    texts_to_process = []
    if hasattr(args, 'source') and args.source and os.path.exists(args.source):
        print(f"Reading texts from source file: {args.source}")
        with open(args.source, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # フォーマット: speaker_id|text もしくは text のみ
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            texts_to_process.append((parts[0], '|'.join(parts[1:])))
                        else:
                            texts_to_process.append((str(args.speaker_id), line))
                    else:
                        texts_to_process.append((str(args.speaker_id), line))
    else:
        # コマンドラインからの単一テキスト
        texts_to_process.append((str(args.speaker_id), args.text))
    
    # 各テキストを処理
    for idx, (speaker_str, text) in enumerate(texts_to_process):
        speaker_id = int(speaker_str) if speaker_str.isdigit() else args.speaker_id
        
        print(f"\n{'='*60}")
        print(f"Processing text {idx+1}/{len(texts_to_process)}")
        print(f"Speaker ID: {speaker_id}")
        print(f"Input Text: {text}")
        
        preprocess_func = get_preprocess_method(preprocess_config)
        sequence = preprocess_func(text, preprocess_config)
        
        real_len = len(sequence)
        print(f"Original Length: {real_len}")
        
        # 1. パディング処理: 常に max_length に揃える
        if real_len > MODEL_MAX_LENGTH:
            print(f"Warning: Text too long ({real_len}). Truncating to {MODEL_MAX_LENGTH}.")
            real_len = MODEL_MAX_LENGTH # Safety limit

        padded_sequence = np.zeros((1, MODEL_MAX_LENGTH), dtype=np.int64)
        padded_sequence[0, :real_len] = sequence[:real_len]

        # 入力変数（引数から制御パラメータを取得）
        texts = padded_sequence
        src_lens = np.array([real_len], dtype=np.int64)
        max_src_len = np.array([MODEL_MAX_LENGTH], dtype=np.int64)
        speakers = np.array([speaker_id], dtype=np.int64)
        p_control = np.array(args.pitch_control, dtype=np.float32)
        e_control = np.array(args.energy_control, dtype=np.float32)
        d_control = np.array(args.duration_control, dtype=np.float32)

        # FastSpeech2推論とHiFi-GAN処理を実行
        _synthesize(fs2_net, hifi_net, texts, src_lens, max_src_len, speakers, 
                   p_control, e_control, d_control, preprocess_config, sequence, real_len, idx)

def _synthesize(fs2_net, hifi_net, texts, src_lens, max_src_len, speakers, 
               p_control, e_control, d_control, preprocess_config, sequence, real_len, idx=0):

    # -------------------------------------------
    # FastSpeech2 推論 (Smart Input & Shape Setting)
    # -------------------------------------------
    print("Running FastSpeech2...")

    # AILIAにシェイプを通知 (エラー回避)
    try:
        fs2_net.set_input_shape(fs2_net.find_blob_index_by_name("texts"), texts.shape)
        fs2_net.set_input_shape(fs2_net.find_blob_index_by_name("src_lens"), (1,))
        fs2_net.set_input_shape(fs2_net.find_blob_index_by_name("max_src_len"), (1,))
    except: pass
    
    inputs = {}
    inputs["texts"] = texts
    inputs["src_lens"] = src_lens
    inputs["max_src_len"] = max_src_len
    inputs["p_control"] = p_control
    inputs["d_control"] = d_control

    # オプション入力のチェック
    try:
        if fs2_net.find_blob_index_by_name("speakers") != -1:
            inputs["speakers"] = speakers
    except: pass
    
    try:
        if fs2_net.find_blob_index_by_name("e_control") != -1:
            inputs["e_control"] = e_control
    except: pass

    # 推論実行
    fs2_res = fs2_net.predict(inputs)
    
    
    # -------------------------------------------
    # 結果のクリーンアップ (spノイズ除去と正確な切り出し)
    # -------------------------------------------
    mel_output_padded = fs2_res[1]
    d_rounded = fs2_res[5] # 各文字の長さ

    # --- [Step 1] sp (無音) 区間の完全ミュート処理 ---
    # sp の ID を特定
    cleaner_name = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sp_id_seq = text_to_sequence("{sp}", cleaner_name)
    sp_id = sp_id_seq[0]

    # Mel全体の最小値（=無音レベル）を取得
    min_mel_val = np.min(mel_output_padded)

    current_frame = 0
    # 入力テキストの各トークンを走査し、spなら塗りつぶす
    for i in range(real_len):
        dur = int(d_rounded[0, i])
        token_id = sequence[i]

        if token_id == sp_id:
            # sp区間を最小値で上書き (ノイズ除去)
            mel_output_padded[0, current_frame : current_frame + dur, :] = min_mel_val
        
        current_frame += dur

    # --- [Step 2] 正確な切り出しとバッファ処理 ---
    # 有効な音声長を計算
    valid_durations = d_rounded[0, :real_len]
    valid_mel_len = int(np.sum(valid_durations))
    print(f"Calculated Mel Length (Content): {valid_mel_len}")

    # ゴミを含まないよう、有効部分だけを切り出す
    mel_output_clean = mel_output_padded[:, :valid_mel_len, :]
    
    # バッファ（余韻）が必要な場合、無音（最小値）を追加
    if MEL_BUFFER_FRAMES > 0:
        silence_padding = np.full(
            (1, MEL_BUFFER_FRAMES, mel_output_clean.shape[2]),
            min_mel_val,
            dtype=mel_output_clean.dtype
        )
        mel_output = np.concatenate([mel_output_clean, silence_padding], axis=1)
        print(f"Added {MEL_BUFFER_FRAMES} frames of silence padding.")
    else:
        mel_output = mel_output_clean
    
    # -------------------------------------------
    # HiFi-GAN 推論
    # -------------------------------------------
    print("Running HiFi-GAN...")
    
    mel_input = np.ascontiguousarray(mel_output.transpose(0, 2, 1))
    
    try:
        hifi_net.set_input_shape(hifi_net.find_blob_index_by_name("mel_input"), mel_input.shape)
    except: pass

    audio_res = hifi_net.predict([mel_input])
    wav = audio_res[0].squeeze()

    # -------------------------------------------
    # 保存
    # -------------------------------------------
    MAX_WAV_VALUE = 32768.0
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype('int16')
    
    # 複数ファイル対応: インデックスを使った保存パス生成
    if idx > 0:
        base, ext = os.path.splitext(args.savepath)
        savepath = f"{base}_{idx}{ext}"
    else:
        savepath = args.savepath
    
    print(f"Saving to {savepath}")
    
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(savepath, sampling_rate, wav)
    print(f"🎉 Saved Audio: {savepath}")

    # Plot saving
    plot_path = savepath.replace(".wav", "_mel.png")
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_output[0].T, aspect="auto", origin="lower")
    plt.title(f"Generated Mel (Len: {mel_output.shape[1]})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"🎉 Saved Plot: {plot_path}")

if __name__ == "__main__":
    infer()