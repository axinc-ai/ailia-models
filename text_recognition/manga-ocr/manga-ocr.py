import sys
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer
import cv2

import ailia
import re
import jaconv
import numpy

# import local modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

IMG_PATH = './demo1.png'

REPO_NAME = 'kha-white/manga-ocr-base'
WEIGHT_DEC_PATH = 'manga-ocr-decoder.onnx'
MODEL_DEC_PATH = 'manga-ocr-decoder.onnx.prototxt'
WEIGHT_ENC_PATH = 'manga-ocr-encoder.onnx'
MODEL_ENC_PATH = 'manga-ocr-encoder.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/manga-ocr'

def preprocess(img, extractor):
    return extractor(img, return_tensors='pt').pixel_values.squeeze()

def postprocess(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[.,]{2,}', lambda x: (x.end() - x.start()) * '.'.text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text

# ======================
# Main functions
# ======================

def get_img_features(enc_net, img):
    img = numpy.array([numpy.array(img, dtype=numpy.float32)])
    features = enc_net.predict(img)
    return features

def inference_logits(dec_net, features, tokenizer):
    input_ids = numpy.ndarray([1,197]).astype(numpy.uint8)
    output =  dec_net.predict([input_ids, features])
    return output

def predict(img, enc_net, dec_net, tokenizer):
    features = get_img_features(enc_net, img)
    
    # input_idsを作る
    tokenizer(features)

    out = inference_logits(dec_net, features)
    tokenizer.decode(out, skip_special_tokens=True)
    return out

def main():
    img = Image.open(IMG_PATH)
    img = img.convert('L').convert('RGB')
    check_and_download_models(MODEL_ENC_PATH, WEIGHT_ENC_PATH, REMOTE_PATH)
    check_and_download_models(MODEL_DEC_PATH, WEIGHT_DEC_PATH, REMOTE_PATH)

    enc_net = ailia.Net('./' + MODEL_ENC_PATH, './' + WEIGHT_ENC_PATH, env_id=0)
    dec_net = ailia.Net('./' + MODEL_DEC_PATH, './' + WEIGHT_DEC_PATH, env_id=0)
    extractor = AutoFeatureExtractor.from_pretrained(REPO_NAME)
    tokenizer = AutoTokenizer.from_pretrained(REPO_NAME)

    ipt = preprocess(img, extractor)
    features = predict(ipt, enc_net, dec_net, tokenizer)
    output = postprocess(tokenizer.decode(features))

if __name__ == '__main__':
    main()
