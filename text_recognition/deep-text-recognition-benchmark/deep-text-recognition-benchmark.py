import sys
import time
import codecs
import argparse

import cv2

import ailia
# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402


# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'None-ResNet-None-CTC.onnx.prototxt'
WEIGHT_PATH = 'None-ResNet-None-CTC.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/'

IMAGE_PATH = 'demo_image/demo_1.png'
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 32
SLEEP_TIME = 0  # for webcam mode


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
args = parser.parse_args()


# ======================
# Utils
# ======================
#def preprocess_image(img):
#    if img.shape[2] == 3:
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#    elif img.shape[2] == 1:
#        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
#    img = cv2.bitwise_not(img)
#    return img


# ======================
# Main functions
# ======================
#def recognize_from_image():
    # prepare input data
#    etl_word = codecs.open(ETL_PATH, 'r', 'utf-8').readlines()
#    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
#    img = preprocess_image(img)

    # net initialize
#    env_id = ailia.get_gpu_environment_id()
#    print(f'env_id: {env_id}')
#    classifier = ailia.Classifier(
#        MODEL_PATH,
#        WEIGHT_PATH,
#        env_id=env_id,
#        format=ailia.NETWORK_IMAGE_FORMAT_GRAY,
#        range=ailia.NETWORK_IMAGE_RANGE_U_FP32
#    )

    # inference
#    print('Start inference...')
#    if args.benchmark:
#        print('BENCHMARK mode')
#        for i in range(5):
#            start = int(round(time.time() * 1000))
#            classifier.compute(img, MAX_CLASS_COUNT)
#            end = int(round(time.time() * 1000))
#            print(f'\tailia processing time {end - start} ms')
#    else:
#        classifier.compute(img, MAX_CLASS_COUNT)

    # get result
#    count = classifier.get_class_count()
#    print(f'class_count: {count}')

#    for idx in range(count):
#        print(f"+ idx={idx}")
#        info = classifier.get_class(idx)
#        print(f"  category={info.category} [ {etl_word[info.category]} ]")
#        print(f"  prob={info.prob}")


#def main():
#    # model files check and download
#    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

#    if args.video is not None:
#        # video mode
#        recognize_from_video()
#    else:
#        # image mode
#        recognize_from_image()


#if __name__ == '__main__':
#    main()

















import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from deep_text_recognition_benchmark_utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition_benchmark_dataset import RawDataset, AlignCollate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys

sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402

MODEL_PATH = 'None-ResNet-None-CTC.onnx.prototxt'
WEIGHT_PATH = 'None-ResNet-None-CTC.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/'


def demo(opt):
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    """ model configuration """
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    imgH = 32
    imgW = 100
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
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            print(image.shape)
            sample = cv2.imread("demo_image/demo_2.jpg")
            sample = cv2.resize(sample,(imgW,imgH),interpolation=cv2.INTER_CUBIC)
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            sample = sample/127.5 - 1.0

            #print(sample.shape)
            #print(image)
            #print(sample)

            # For max length prediction
            #length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
            #text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            import onnxruntime
            import numpy
            session = onnxruntime.InferenceSession("None-ResNet-None-CTC.onnx")
            session.get_modelmeta()
            first_input_name = session.get_inputs()[0].name
            print(first_input_name)
            preds = session.run([], {first_input_name: image.to('cpu').detach().numpy().copy()})
            preds = preds[0]
            print(preds.shape)
            preds = torch.from_numpy(preds.astype(numpy.float32)).clone()

            import ailia
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
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                #if 'Attn' in opt.Prediction:
                #    pred_EOS = pred.find('[s]')
                #    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                #    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default="demo_image/", help='path to image_folder which contains text images')
    #parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    #parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    #parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    #parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    #parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    #parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    #parser.add_argument('--rgb', action='store_true', help='use rgb input')
    #parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    #parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    #parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    #parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    #parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    #parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    #parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    #parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    #parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    #parser.add_argument('--output_channel', type=int, default=512,
    #                    help='the number of output channel of Feature extractor')
    #parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    #if opt.sensitive:
    #    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
