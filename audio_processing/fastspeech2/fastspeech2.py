import ailia
import numpy as np
import yaml
import sys
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wavfile
from g2p_en import G2p
from text import text_to_sequence
import re

# utilsのインポートのためにパスを追加
sys.path.append(".")

# ===========================
# 1. 設定
# ===========================

#　今後paeser_argsで外部から指定できるようにする
FS2_ONNX_PATH = "./onnx/fastspeech2/ljspeech.onnx"
HIFI_ONNX_PATH = "./onnx/hifigan/hifigan.onnx"
PREPROCESS_CONFIG = "config/LJSpeech/preprocess.yaml"
OUTPUT_DIR = "onnx/result/LJSpeech"

# ★重要: エクスポート時と同じ最大長 (VRAM不足回避のため 600 で統一)
MODEL_MAX_LENGTH = 600

# 音声途切れを防ぐためのバッファ (10フレーム ≈ 0.1秒)
MEL_BUFFER_FRAMES = 40

TEXT_TO_SPEAK = "Ailia SDK makes it easy to deploy deep learning models. This script handles both single and multi speaker models automatically."
#TEXT_TO_SPEAK = "早上好。我们目前正在对机器学习模型进行修改。"

SPEAKER_ID = 0

# ===========================
# 2. 前処理
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

# ===========================
# 3. 推論メイン
# ===========================
def infer():
    if not os.path.exists(FS2_ONNX_PATH) or not os.path.exists(HIFI_ONNX_PATH):
        print("Error: ONNX file not found.")
        return

    print("Loading Config...")
    preprocess_config = yaml.load(open(PREPROCESS_CONFIG, "r"), Loader=yaml.FullLoader)
    
    print("Loading ONNX Models...")
    env_id = ailia.get_gpu_environment_id()
    
    fs2_net = ailia.Net(None, FS2_ONNX_PATH, env_id=env_id)
    hifi_net = ailia.Net(None, HIFI_ONNX_PATH, env_id=env_id)

    # -------------------------------------------
    # 入力データの準備（パディング処理）
    # -------------------------------------------
    print(f"Input Text: {TEXT_TO_SPEAK}")
    sequence = preprocess_english(TEXT_TO_SPEAK, preprocess_config)
    real_len = len(sequence)
    print(f"Original Length: {real_len}")
    
    # 1. パディング処理: 常に max_length に揃える
    if real_len > MODEL_MAX_LENGTH:
        print(f"Warning: Text too long ({real_len}). Truncating to {MODEL_MAX_LENGTH}.")
        real_len = MODEL_MAX_LENGTH # Safety limit

    padded_sequence = np.zeros((1, MODEL_MAX_LENGTH), dtype=np.int64)
    padded_sequence[0, :real_len] = sequence[:real_len]

    # 入力変数
    texts = padded_sequence
    src_lens = np.array([real_len], dtype=np.int64)
    max_src_len = np.array([MODEL_MAX_LENGTH], dtype=np.int64)
    speakers = np.array([SPEAKER_ID], dtype=np.int64)
    p_control = np.array(1.0, dtype=np.float32)
    e_control = np.array(1.0, dtype=np.float32)
    d_control = np.array(1.0, dtype=np.float32)

    # -------------------------------------------
    # FastSpeech2 推論 (Smart Input & Shape Setting)
    # -------------------------------------------
    print("Running FastSpeech2...")

    # AILIAにシェイプを通知 (エラー回避)
    try:
        # textsの形状を実サイズで通知し、内部バッファを調整させる
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

    # オプション入力のチェック (存在すれば追加)
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
    # 結果の切り出し (正確なトリミング)
    # -------------------------------------------
    mel_output_padded = fs2_res[1]
    d_rounded = fs2_res[5] # 各文字の長さ (d_rounded)

    # 1. 有効な文字数分の長さ（durations for real text）を合計
    valid_durations = d_rounded[0, :real_len]
    
    # 2. 合計フレーム数を計算し、バッファを追加 (途切れ防止)
    # [修正ポイント]
    valid_mel_len = int(np.sum(valid_durations)) + MEL_BUFFER_FRAMES
    
    print(f"Calculated Mel Length (with buffer): {valid_mel_len}")

    # 3. 有効な部分だけスパッと切り落とす
    mel_output = mel_output_padded[:, :valid_mel_len, :]
    
    # -------------------------------------------
    # HiFi-GAN 推論
    # -------------------------------------------
    print("Running HiFi-GAN...")
    
    mel_input = mel_output.transpose(0, 2, 1)
    
    # HiFi-GANのシェイプも通知 (安定性向上)
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
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    wav_path = os.path.join(OUTPUT_DIR, "output.wav")
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(wav_path, sampling_rate, wav)
    print(f"🎉 Saved Audio: {wav_path}")

    plot_path = os.path.join(OUTPUT_DIR, "output_mel.png")
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_output[0].T, aspect="auto", origin="lower")
    plt.title(f"Generated Mel (Len: {valid_mel_len})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"🎉 Saved Plot: {plot_path}")

if __name__ == "__main__":
    infer()