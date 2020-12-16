import sys
import time
import codecs
import argparse
import numpy

import cv2

import ailia
# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402

import string

import torch
import torch.utils.data
import torch.nn.functional as F

from deep_text_recognition_benchmark_utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition_benchmark_dataset import RawDataset, AlignCollate


# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'None-ResNet-None-CTC.onnx.prototxt'
WEIGHT_PATH = 'None-ResNet-None-CTC.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/'

IMAGE_PATH = 'demo_image/demo_1.png'
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 32


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='deep text recognition benchmark.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGEFILE_PATH',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-o', '--onnx',
    action='store_true',
    help='Use onnx runtime'
)
args = parser.parse_args()


# ======================
# Utils
# ======================
def preprocess_image(sample):
    sample = cv2.resize(sample,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_CUBIC)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample = sample/127.5 - 1.0
    return sample

def recognize_from_image(opt):
    """ model configuration """
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    imgH = IMAGE_HEIGHT
    imgW = IMAGE_WIDTH
    PAD = False
    batch_size = 10
    workers = 4
    batch_max_length = 25

    converter = CTCLabelConverter(character)
    opt.num_class = len(converter.character)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=batch_size,
        shuffle=False,
        num_workers=int(workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    device="cpu"
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            print(image.shape)
            sample = cv2.imread("demo_image/demo_2.jpg")
            sample = preprocess_image(sample)

            if args.onnx:
                import onnxruntime
                session = onnxruntime.InferenceSession("None-ResNet-None-CTC.onnx")
                session.get_modelmeta()
                first_input_name = session.get_inputs()[0].name
                print(first_input_name)
                preds = session.run([], {first_input_name: image.to('cpu').detach().numpy().copy()})
                preds = preds[0]
                print(preds.shape)
                preds = torch.from_numpy(preds.astype(numpy.float32)).clone()
            else:
                env_id = ailia.get_gpu_environment_id()
                net = ailia.Net("None-ResNet-None-CTC.onnx.prototxt","None-ResNet-None-CTC.onnx",env_id=env_id)
                input_img = image.to('cpu').detach().numpy().copy()
                input_img[0,:,:,:] = sample
                print(input_img.shape)
                preds = net.predict(input_img)
                print(preds.shape)
                preds = torch.from_numpy(preds.astype(numpy.float32)).clone()

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)

            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')

if __name__ == '__main__':
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default="demo_image/", help='path to image_folder which contains text images')
    opt = parser.parse_args()

    #cudnn.benchmark = True
    #cudnn.deterministic = True
    #opt.num_gpu = torch.cuda.device_count()

    recognize_from_image(opt)
