import sys
import time
import argparse

import numpy as np
import cv2

import ailia

sys.path.append('../util')
from model_utils import check_and_download_models  # noqa: E402


# PYTORCHGAN-ZOO
# SOFTWARE LICENSE AGREEMENT
# https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/LICENSE
#
# Copyright 2019, Facebook
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# ======================
# Parameters
# ======================
SAVE_IMAGE_PATH = 'output.png'

MODEL_PATH = 'pytorch-gnet-animeface.onnx.prototxt'
WEIGHT_PATH = 'pytorch-gnet-animeface.onnx'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/pytorch-gan/'

# =======================
# Arguments Parser Config
# =======================
parser = argparse.ArgumentParser(
    description='Generation of anime character faces using a GNet trained from the PytorchZoo GAN.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance.'
)
args = parser.parse_args()

def generate_image():
    # prepare input data
    rand_input = np.random.rand(1,512).astype(np.float32)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    gnet = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            _ = gnet.predict(rand_input)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        _ = gnet.predict(rand_input)
        
    # postprocessing

    [output_blob_idx] = gnet.get_output_blob_list()
    output_data = gnet.get_blob_data(output_blob_idx)

    outp = np.clip((0.5 + 255*output_data.transpose((2,3,1,0)).reshape((64,64,3))).astype(np.float32),0,255)

    ## If using PIL instead of OpenCV:
    #from PIL import Image
    #img = Image.fromarray(outp.astype(np.uint8),'RGB')
    #img.save(args.savepath)

    cv2.imwrite(args.savepath, cv2.cvtColor(outp.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    generate_image()


if __name__ == '__main__':
    main()
