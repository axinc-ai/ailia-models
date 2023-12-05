NUM_LAYERS = 12
NUM_HEAD = 16
N_DIM = 1024
PREFIX_MODE = 1
NUM_QUANTIZERS = 8
SAMPLE_RATE = 24000

lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
    'mix': "",
}

lang2code = {
    'zh': 0,
    'ja': 1,
    "en": 2,
}

token2lang = {
    '[ZH]': "zh",
    '[JA]': "ja",
    "[EN]": "en",
    "": "mix"
}

code2lang = {
    0: 'zh',
    1: 'ja',
    2: "en",
}

langdropdown2token = {
    'English': "[EN]",
    '中文': "[ZH]",
    '日本語': "[JA]",
    'Mix': "",
}

# Text
NUM_TEXT_TOKENS = 2048

# Audio
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins
NUM_MEL_BINS = 100  # BigVGAN bigvgan_24khz_100band

# Speaker
NUM_SPEAKER_CLASSES = 4096
SPEAKER_EMBEDDING_DIM = 64
