
import sys
import time
from logging import getLogger

from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

import numpy as np

# import original modules
sys.path.append("../../util")

from model_utils import check_and_download_models, check_and_download_file  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WAV_PATH = "example/ja.mp3"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Whisper", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio")
#parser.add_argument(
#    "--intermediate", action="store_true", help="display intermediate state."
#)
#parser.add_argument(
#    "--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
#)
args = update_parser(parser)


# ======================
# Models
# ======================

WEIGHT_PATH = "sensevoice_small.onnx"
MODEL_PATH = "sensevoice_small.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sensevoice/"

def recognize_from_audio():
	# input audio loop
	for audio_path in args.input:
		logger.info(audio_path)

		model = SenseVoiceSmall(model_dir="./", batch_size=10, quantize=False, cache_dir="./", env_id=args.env_id)

		# inference
		wav_or_scp = [audio_path]
		#wav_or_scp = ["./example/en.mp3", "./example/ja.mp3", "./example/ax.wav"]

		import time

		start = int(round(time.time() * 1000))
		res = model(wav_or_scp, language="auto", use_itn=True)
		end = int(round(time.time() * 1000))
		estimation_time = end - start
		#logger.info(f"\tencoder processing time {estimation_time} ms")
		print(f"\tprocessing time {estimation_time} ms")

		print([rich_transcription_postprocess(i) for i in res])

		# inference
		logger.info("Start inference...")

	logger.info("Script finished successfully.")

def main():
	check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
	
	recognize_from_audio()

if __name__ == "__main__":
	main()
