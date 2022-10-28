import sys
import time
import threading
import multiprocessing as mp
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def capture_microphone(que, ready, pause, fin, sample_rate, speaker=False):
    import soundcard as sc

    THRES_SPEECH_POW = 0.001
    THRES_SILENCE_POW = 0.0001
    INTERVAL = sample_rate * 3
    INTERVAL_MIN = sample_rate * 1.5
    BUFFER_MAX = sample_rate * 10
    v = np.ones(100) / 100
    try:
        ready.set()

        def send(audio, n):
            if INTERVAL_MIN < n:
                que.put_nowait(audio[:n])

        # start recording
        mic_id = str(sc.default_speaker().name) if speaker else str(sc.default_microphone().name)
        buf = np.array([], dtype=np.float32)
        with sc.get_microphone(id=mic_id, include_loopback=speaker).recorder(
                samplerate=sample_rate, channels=1) as mic:
            while not fin.is_set():
                if pause.is_set():
                    buf = buf[:0]
                    time.sleep(0.1)
                    continue

                audio = mic.record(INTERVAL)
                audio = audio.reshape(-1)
                square = audio ** 2
                if np.max(square) >= THRES_SPEECH_POW:
                    sys.stdout.write(".")
                    sys.stdout.flush()

                    # 平準化
                    conv = np.convolve(square, v, 'valid')
                    conv = np.pad(conv, (0, len(v) - 1), mode='edge')
                    # 0.1s刻みで0.5s区間をチェック
                    s = sample_rate // 10
                    x = [(min(i + s * 5, INTERVAL), np.any(conv[i:i + s * 5] >= THRES_SILENCE_POW))
                         for i in range(0, INTERVAL - 5 * s + 1, s)]

                    # Speech section
                    speech = [a[0] for a in x if a[1]]
                    if speech:
                        if len(buf) == 0 and 1 < len(speech):
                            i = max(speech[0] - 3 * s, 0)
                            audio = audio[i:]
                        i = speech[-1]
                        audio = audio[:i]
                    else:
                        i = 0

                    buf = np.concatenate([buf, audio])
                    if i < INTERVAL:
                        send(buf, len(buf))
                        buf = buf[:0]
                    elif BUFFER_MAX < len(buf):
                        i = np.argmin(buf[::-1])
                        i = len(buf) - i
                        if 0 < i:
                            send(buf, i)
                            buf = buf[i:]
                        else:
                            send(buf, len(buf))
                            buf = buf[:0]
                elif 0 < len(buf):
                    send(buf, len(buf))
                    buf = buf[:0]
            pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(e)


def start_microphone_input(sample_rate, speaker=False, thread=False, queue_size=2):
    que = mp.Queue(maxsize=queue_size)
    ready = mp.Event()
    pause = mp.Event()
    fin = mp.Event()

    if thread:
        p = threading.Thread(
            target=capture_microphone,
            args=(que, ready, pause, fin, sample_rate, speaker),
            daemon=True)
    else:
        p = mp.Process(
            target=capture_microphone,
            args=(que, ready, pause, fin, sample_rate, speaker),
            daemon=True)
    p.start()

    # キャプチャスレッド起動待ち
    while p.is_alive():
        if ready.is_set():
            break

    if not p.is_alive():
        raise Exception('Fail to start microphone capture.')

    params = dict(
        p=p,
        que=que,
        pause=pause,
        fin=fin,
    )

    return params
