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
import onnx

# ===========================
# Settings
# ===========================

# リポジトリのルートにあるutilsを参照できるようにする
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa

# モデル設定
WEIGHT_PATH_FS2 = 'ljspeech.onnx'
MODEL_PATH_FS2 = 'ljspeech.onnx.prototxt'
WEIGHT_PATH_HIFI = 'hifigan.onnx'
MODEL_PATH_HIFI = 'hifigan.onnx.prototxt'
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/fastspeech2/"

PREPROCESS_CONFIG = "config/LJSpeech/preprocess.yaml"

# ★重要: エクスポート時と同じ最大長 (VRAM不足回避のため 600 で統一)
MODEL_MAX_LENGTH = 600

# ===========================
# Arguments
# ===========================
parser = get_base_parser(
    'FastSpeech2 (Ailia Inference)',
     None,
    'output_ailia.wav'
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
    
    # ONNXモデルの出力名を取得（ailiaSDKでは直接取得できないため、ONNXファイルから読み込む）
    onnx_model = onnx.load(args.onnx_fs2)
    fs2_output_names = [output.name for output in onnx_model.graph.output]
    fs2_input_names = [inp.name for inp in onnx_model.graph.input 
                       if inp.name not in [n.name for n in onnx_model.graph.initializer]]
    
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
        
        # max_src_lenの形状を確認して適切に設定
        max_src_len = None
        if "max_src_len" in fs2_input_names:
            for inp in onnx_model.graph.input:
                if inp.name == "max_src_len":
                    if hasattr(inp.type, 'tensor_type') and hasattr(inp.type.tensor_type, 'shape'):
                        max_src_len_shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                                             for d in inp.type.tensor_type.shape.dim]
                        if len(max_src_len_shape) == 0:
                            # スカラーとして渡す
                            max_src_len = np.array(MODEL_MAX_LENGTH, dtype=np.int64)
                        else:
                            # 配列として渡す（通常は[1]）
                            max_src_len = np.array([MODEL_MAX_LENGTH], dtype=np.int64)
                    else:
                        # デフォルトで配列として渡す（LJSpeech互換性のため）
                        max_src_len = np.array([MODEL_MAX_LENGTH], dtype=np.int64)
                    break
        
        # speakersの形状を確認して適切に設定
        if "speakers" in fs2_input_names:
            # ONNXモデルの実際の入力形状を確認
            for inp in onnx_model.graph.input:
                if inp.name == "speakers":
                    if hasattr(inp.type, 'tensor_type') and hasattr(inp.type.tensor_type, 'shape'):
                        speakers_shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                                         for d in inp.type.tensor_type.shape.dim]
                        if len(speakers_shape) == 2 and (speakers_shape[1] == 1 or speakers_shape[1] == 'batch_size'):
                            # (batch, 1) の形状
                            speakers = np.array([[speaker_id]], dtype=np.int64)
                        else:
                            # (batch,) の形状
                            speakers = np.array([speaker_id], dtype=np.int64)
                    else:
                        # デフォルトで (batch, 1) を試す
                        speakers = np.array([[speaker_id]], dtype=np.int64)
                    break
        else:
            speakers = None
        
        p_control = np.array(args.pitch_control, dtype=np.float32)
        e_control = np.array(args.energy_control, dtype=np.float32)
        d_control = np.array(args.duration_control, dtype=np.float32)

        # FastSpeech2推論とHiFi-GAN処理を実行
        _synthesize(fs2_net, hifi_net, texts, src_lens, max_src_len, speakers, 
                   p_control, e_control, d_control, preprocess_config, sequence, real_len, idx, 
                   fs2_output_names, fs2_input_names)

def _synthesize(fs2_net, hifi_net, texts, src_lens, max_src_len, speakers, 
               p_control, e_control, d_control, preprocess_config, sequence, real_len, idx, 
               fs2_output_names, fs2_input_names):

    # -------------------------------------------
    # FastSpeech2 推論
    # -------------------------------------------
    print("Running FastSpeech2...")

    inputs = {}
    inputs["texts"] = texts
    inputs["src_lens"] = src_lens
    if max_src_len is not None:
        inputs["max_src_len"] = max_src_len
    
    # control変数はスカラーまたは配列のどちらでも動作するように
    if "p_control" in fs2_input_names:
        inputs["p_control"] = p_control
    if "d_control" in fs2_input_names:
        inputs["d_control"] = d_control
    if "e_control" in fs2_input_names:
        inputs["e_control"] = e_control
    
    if speakers is not None:
        inputs["speakers"] = speakers

    # 入力形状のデバッグ情報
    print("\n=== FastSpeech2 Input Shapes ===")
    for k, v in inputs.items():
        print(f"  {k:20s}: {v.shape if hasattr(v, 'shape') else type(v)}")
    print("=" * 40)

    try:
        fs2_res = fs2_net.predict(inputs)
    except Exception as e:
        print(f"❌ FastSpeech2 inference failed: {e}")
        return
    
    # -------------------------------------------
    # 結果の切り出し
    # -------------------------------------------
    try:
        d_rounded_index = fs2_output_names.index("d_rounded")
        postnet_index = fs2_output_names.index("postnet_output")
    except ValueError:
        d_rounded_index = 5
        postnet_index = 1
        
    mel_output_whole = fs2_res[postnet_index] # [1, MaxLen, 80]
    d_rounded = fs2_res[d_rounded_index]      # [1, MaxLen]

    # 元のリポジトリと同じ処理：mel_lenを計算（synth_samplesと同様）
    valid_durations = d_rounded[0, :real_len]
    mel_len = int(np.sum(valid_durations))
    
    print(f"Generated Mel Length: {mel_len}")
    
    # 元のリポジトリと同じ処理：mel_lenで切り出す（バッファなし）
    mel_output = mel_output_whole[:, :mel_len, :]

    # -------------------------------------------
    # HiFi-GAN 推論（元のリポジトリと同じ処理）
    # -------------------------------------------
    print("Running HiFi-GAN...")
    
    # 元のリポジトリと同じ処理：[1, MelLen, 80] -> [1, 80, MelLen]
    # synth_samplesでは predictions[1].transpose(1, 2) を使用
    mel_input = mel_output.transpose(0, 2, 1).astype(np.float32)
    
    # HiFi-GANのONNXモデルは固定長（3000フレーム）を期待しているため、パディングが必要
    # ただし、元のリポジトリの処理に近づけるため、シンプルなパディングを使用
    HIFI_FIXED_LENGTH = 3000
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    actual_mel_len = mel_input.shape[2]
    
    if actual_mel_len < HIFI_FIXED_LENGTH:
        # 元のリポジトリに近い処理：最後のフレームを繰り返してパディング
        pad_length = HIFI_FIXED_LENGTH - actual_mel_len
        last_frame = mel_input[:, :, -1:]
        padding = np.repeat(last_frame, pad_length, axis=2)
        mel_input = np.concatenate([mel_input, padding], axis=2)
        print(f"Padded mel_input from {actual_mel_len} to {HIFI_FIXED_LENGTH} frames")
    elif actual_mel_len > HIFI_FIXED_LENGTH:
        # 3000フレームを超える場合は切り詰め
        mel_input = mel_input[:, :, :HIFI_FIXED_LENGTH]
        actual_mel_len = HIFI_FIXED_LENGTH
        print(f"Truncated mel_input from {actual_mel_len} to {HIFI_FIXED_LENGTH} frames")
    
    try:
        audio_res = hifi_net.predict([mel_input])
        wav = audio_res[0].squeeze()
    except Exception as e:
        print(f"❌ HiFi-GAN inference failed: {e}")
        return
    
    # 元のリポジトリと同じ処理：lengths = mel_len * hop_length で切り出す
    # vocoder_inferでは lengths[i] で切り出している
    audio_len = mel_len * hop_length
    if len(wav) > audio_len:
        wav = wav[:audio_len]
        print(f"Trimmed audio to {audio_len} samples (mel_len={mel_len} * hop_length={hop_length})")

    # -------------------------------------------
    # 保存（元のリポジトリと同じ処理）
    # -------------------------------------------
    # 元のリポジトリのvocoder_inferと同じ処理：
    # wavs = wavs.cpu().numpy() * max_wav_value
    MAX_WAV_VALUE = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
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

