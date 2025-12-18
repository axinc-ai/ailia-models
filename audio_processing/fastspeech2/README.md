# FastSpeech2

## Input

[FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)

## Output

Audio file (`output.wav`) and Mel-spectrogram plot (`output_mel.png`).

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

### Single-sentence Synthesis

For the sample text (default):

```bash
$ python3 fastspeech2.py
```

Specify your own text:

```bash
$ python3 fastspeech2.py --text "Hello, this is a test."
```

Specify speaker ID for multi-speaker models:

```bash
$ python3 fastspeech2.py \
  --text "Hello world" \
  --speaker_id 0
```

Control pitch, energy, and speaking rate:

```bash
$ python3 fastspeech2.py \
  --text "Hello world" \
  --pitch_control 1.2 \
  --energy_control 1.1 \
  --duration_control 0.8
```

### Batch Synthesis

Synthesize from a text file (like train.txt or val.txt format):

```bash
$ python3 fastspeech2.py --source input.txt --mode batch
```

The source file format should be like:
```
basename1|speaker_id|text
basename2|speaker_id|text
```

Or simplified format (for single speaker):
```
This is the first sentence.
This is the second sentence.
```

### Multi-Speaker Models

For LibriTTS (English, Multi-Speaker)

```bash
$ python3 fastspeech2.py \
  --text "Hello, I am speaking from a multi-speaker model." \
  --preprocess_config config/LibriTTS/preprocess.yaml \
  --onnx_fs2 onnx/fastspeech2/libritts.onnx \
  --speaker_id 0
```

For AISHELL-3 (Mandarin, Multi-Speaker): //実行できない

```bash
$ python3 fastspeech2.py \
  --text "你好" \
  --preprocess_config config/AISHELL3/preprocess.yaml \
  --onnx_fs2 onnx/fastspeech2/aishell3.onnx \
  --speaker_id 16
```

## Options

### Core Arguments (same as original FastSpeech2 repo)

- `--source`: Path to a source file with format like train.txt and val.txt (for batch mode)
- `--restore_step`: Step for checkpoint to restore (default: 900000)
- `--mode`: Synthesize mode - 'single' or 'batch' (default: 'single')
- `--text`: Raw text to synthesize (for single-sentence mode only)
- `--speaker_id`: Speaker ID for multi-speaker synthesis (for single-sentence mode only, default: 0)
- `-p`, `--pitch_control`: Control the pitch of the whole utterance, larger value for higher pitch (default: 1.0)
- `-e`, `--energy_control`: Control the energy of the whole utterance, larger value for larger volume (default: 1.0)
- `-d`, `--duration_control`: Control the speed of the whole utterance, larger value for slower speaking rate (default: 1.0)

### Additional Arguments (ailia-specific)

- `--preprocess_config`: Path to preprocess.yaml (default: config/LJSpeech/preprocess.yaml)
- `--model_config`: Path to model.yaml (default: config/LJSpeech/model.yaml)
- `--onnx_fs2`: Path to FastSpeech2 ONNX file
- `--onnx_hifi`: Path to HiFi-GAN ONNX file
- `-s`, `--savepath`: Save path for the output audio file (default: output.wav)
- `-b`, `--benchmark`: Running the inference on the same input 5 times to measure execution performance
- `--env_id`: The backend environment id

## Model

- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

## Requirements

- ailia SDK
- g2p_en
