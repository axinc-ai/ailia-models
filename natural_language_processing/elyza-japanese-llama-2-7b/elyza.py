import time
import sys
import os
from transformers import  AutoTokenizer
from utils_elyza import *
import numpy
import platform

#from utils_rinna_gpt2 import *
import ailia

sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, check_and_download_file  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# PARAMETERS
# ======================
WEIGHT_PATH = "decoder_model.onnx"
MODEL_PATH = "decoder_model.onnx.prototxt"
WEIGHT_PB_PATH = "decoder_model.onnx_data"
tokenizer_path= "elyza/ELYZA-japanese-Llama-2-7b-instruct"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/elyza-japanese-llama-2-7b/"
SAVE_PATH="output.txt"


DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
DEFAULT_TEXT =  "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。"



# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("elyza text generation", None, SAVE_PATH)
# overwrite
parser.add_argument(
    "--input", "-i", default=DEFAULT_TEXT, 
    help="input text"
)
parser.add_argument(
    "--outlength", "-o", default=256,
    help="number of tokens to generate"
)
parser.add_argument(
    "--onnx",
    action="store_true",
    help="By default, the ailia SDK is used, but with this option, you can switch to using ONNX Runtime"
)
args = update_parser(parser, check_input_type=False)





# ======================
# Main function
# ======================
def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_file(WEIGHT_PB_PATH, REMOTE_PATH)

    pf = platform.system()
    if pf == "Darwin":
        logger.info("This model not optimized for macOS GPU currently. So we will use BLAS (env_id = 1).")
        args.env_id = 1

    if args.onnx:
        import onnxruntime
        # Create ONNX Runtime session with GPU as the execution provider
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        # Specify the execution provider (CUDAExecutionProvider for GPU)
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
        ailia_model = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers, sess_options=options)
    else:
        memory_mode = ailia.get_memory_mode(True, True, False, True)
        ailia_model = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #Generate prompt
    logger.info("Input : "+args.input)
    input_promt=generate_prompt(tokenizer, DEFAULT_SYSTEM_PROMPT, args.input)
    

    # inference
    print("Start of text generation")
    if args.benchmark:
        logger.info("BENCHMARK mode")
        for i in range(5):
            start = int(round(time.time() * 1000))
            output = generate_text(tokenizer, ailia_model, input_promt, int(args.outlength), args.onnx)
            end = int(round(time.time() * 1000))
            logger.info("\tailia processing time {} ms".format(end - start))
    else:
        
        output = generate_text(tokenizer, ailia_model, input_promt, int(args.outlength), args.onnx)

    logger.info("output : "+output)
    with open(SAVE_PATH, "w") as fo:
        fo.write(output)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    main() 

