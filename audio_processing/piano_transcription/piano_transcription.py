import sys
import time

import numpy as np
import librosa
import audioread

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

from piano_transcription_utils import RegressionPostProcessor
from piano_transcription_utils import write_events_to_midi

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_NOTE_PATH = 'note_model.onnx'
MODEL_NOTE_PATH = 'note_model.onnx.prototxt'
WEIGHT_PEDAL_PATH = 'pedal_model.onnx'
MODEL_PEDAL_PATH = 'pedal_model.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/piano_transcription/'

AUDIO_PATH = "cut_liszt.mp3"
SAVE_PATH = 'output.mid'

# Audio
SAMPLING_RATE = 16000

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Piano transcription', AUDIO_PATH, SAVE_PATH, input_ftype='audio'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================

def load_audio(
        path, sr=22050, mono=True, offset=0.0, duration=None,
        dtype=np.float32, res_type='kaiser_best',
        backends=[audioread.ffdec.FFmpegAudioFile]):
    """Load audio. Copied from librosa.core.load() except that ffmpeg backend is
    always used in this function."""

    y = []
    with audioread.audio_open(path, backends=backends) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0
        for frame in input_file:
            frame = librosa.core.audio.util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = librosa.core.audio.to_mono(y)

        if sr is not None:
            y = librosa.core.audio.resample(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


def deframe(x):
    """ Deframe predicted segments to original sequence.
    Args:
      x: (N, segment_frames, classes_num)
    Returns:
      y: (audio_frames, classes_num)
    """
    segment_samples = SAMPLING_RATE * 10

    if x.shape[0] == 1:
        return x[0]
    else:
        x = x[:, 0: -1, :]
        """Remove an extra frame in the end of each segment caused by the
        'center=True' argument when calculating spectrogram."""
        (N, segment_samples, classes_num) = x.shape

        y = [x[0, 0: int(segment_samples * 0.75)]]
        for i in range(1, N - 1):
            y.append(x[i, int(segment_samples * 0.25): int(segment_samples * 0.75)])
        y.append(x[-1, int(segment_samples * 0.25):])
        y = np.concatenate(y, axis=0)

        return y


# ======================
# Main functions
# ======================

def preprocess(audio):
    segment_samples = SAMPLING_RATE * 10

    audio = audio[None, :]  # (1, audio_samples)

    # Pad audio to be evenly divided by segment_samples
    audio_len = audio.shape[1]
    pad_len = int(np.ceil(audio_len / segment_samples)) * segment_samples - audio_len

    audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

    # Enframe long sequence to short segments.
    batch = []
    pointer = 0
    while pointer + segment_samples <= audio.shape[1]:
        batch.append(audio[:, pointer: pointer + segment_samples])
        pointer += segment_samples // 2

    batch = np.concatenate(batch, axis=0)
    batch = batch.astype(np.float32)

    return batch


def post_processing(output_dict):
    frames_per_second = 100
    classes_num = 88
    onset_threshold = 0.3
    offset_threshod = 0.3
    frame_threshold = 0.1
    pedal_offset_threshold = 0.2

    post_processor = RegressionPostProcessor(
        frames_per_second,
        classes_num=classes_num, onset_threshold=onset_threshold,
        offset_threshold=offset_threshod,
        frame_threshold=frame_threshold,
        pedal_offset_threshold=pedal_offset_threshold)

    # Post process output_dict to MIDI events
    (est_note_events, est_pedal_events) = \
        post_processor.output_dict_to_midi_events(output_dict)

    return est_note_events, est_pedal_events


def predict(net_note, net_pedal, audio):
    audio_len = audio.shape[0]

    segments = preprocess(audio)

    output_dict = {
        'reg_onset_output': [],
        'reg_offset_output': [],
        'frame_output': [],
        'velocity_output': [],
        'reg_pedal_onset_output': [],
        'reg_pedal_offset_output': [],
        'pedal_frame_output': [],
    }
    pointer = 0
    while True:
        if pointer >= len(segments):
            break

        batch_waveform = segments[pointer: pointer + 1]
        pointer += 1

        # feedforward
        if not args.onnx:
            output = net_note.predict([batch_waveform])
        else:
            output = net_note.run(None, {'input': batch_waveform})

        reg_onset_output, reg_offset_output, frame_output, velocity_output = output

        # feedforward
        if not args.onnx:
            output = net_pedal.predict([batch_waveform])
        else:
            output = net_pedal.run(None, {'input': batch_waveform})

        reg_pedal_onset_output, reg_pedal_offset_output, pedal_frame_output = output

        output_dict['reg_onset_output'].append(reg_onset_output)
        output_dict['reg_offset_output'].append(reg_offset_output)
        output_dict['frame_output'].append(frame_output)
        output_dict['velocity_output'].append(velocity_output)
        output_dict['reg_pedal_onset_output'].append(reg_pedal_onset_output)
        output_dict['reg_pedal_offset_output'].append(reg_pedal_offset_output)
        output_dict['pedal_frame_output'].append(pedal_frame_output)

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    # Deframe to original length
    for key in output_dict.keys():
        output_dict[key] = deframe(output_dict[key])[0: audio_len]

    return output_dict


def audio_recognition(net_note, net_pedal):
    for audio_path in args.input:
        logger.info(audio_path)

        # Load audio
        (audio, _) = load_audio(audio_path, sr=SAMPLING_RATE, mono=True)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output_dict = predict(net_note, net_pedal, audio)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Loggin
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output_dict = predict(net_note, net_pedal, audio)

        est_note_events, est_pedal_events = post_processing(output_dict)

        midi_path = get_savepath(args.savepath, audio_path, ext='.mid')
        logger.info(f'saved at : {midi_path}')
        write_events_to_midi(
            start_time=0, note_events=est_note_events,
            pedal_events=est_pedal_events, midi_path=midi_path)

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('Checking note model...')
    check_and_download_models(WEIGHT_NOTE_PATH, MODEL_NOTE_PATH, REMOTE_PATH)
    logger.info('Checking pedal model...')
    check_and_download_models(WEIGHT_PEDAL_PATH, MODEL_PEDAL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net_note = ailia.Net(MODEL_NOTE_PATH, WEIGHT_NOTE_PATH, env_id=env_id)
        net_pedal = ailia.Net(MODEL_PEDAL_PATH, WEIGHT_PEDAL_PATH, env_id=env_id)
    else:
        import onnxruntime
        net_note = onnxruntime.InferenceSession(WEIGHT_NOTE_PATH)
        net_pedal = onnxruntime.InferenceSession(WEIGHT_PEDAL_PATH)

    audio_recognition(net_note, net_pedal)


if __name__ == '__main__':
    main()
