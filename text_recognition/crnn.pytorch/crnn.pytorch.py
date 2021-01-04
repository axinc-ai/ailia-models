import sys
import time

import cv2
import numpy as np
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'crnn_pytorch.onnx'
MODEL_PATH = 'crnn_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/crnn_pytorch/'

IMAGE_PATH = 'demo.png'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Convolutional Recurrent Neural Network',
    IMAGE_PATH,
    None,
)
parser.add_argument(
    '-o', '--onnx',
    action='store_true',
    default=False,
    help='Use onnx runtime'
)
args = update_parser(parser)

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

# ======================
# Utils
# ======================
def pre_process(image):
    image = np.round((np.array(image.resize((100, 32), Image.BILINEAR))/255 - 0.5)/0.5, 4)
    image = np.expand_dims(np.expand_dims(image, 0), 0).astype(np.float32)
    return image


def post_process(preds, length, alphabet):
    preds = np.argmax(preds, axis=2).transpose(1, 0)[0]
    
    alphabet = alphabet + '-'  # for `-1` index
    dict = {}
    for i, char in enumerate(alphabet):
        dict[char] = i + 1

    assert len(preds)== length, "text with length: {} does not match declared length: {}".format(len(preds), length)
    char_list = []
    for i in range(length):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            char_list.append(alphabet[preds[i] - 1])
    return ''.join(char_list)


def predict(net, image):
    if not args.onnx:
        preds = net.predict({'input.1':image})  # Not yet tried
    else:
        first_input_name = net.get_inputs()[0].name
        first_output_name = net.get_outputs()[0].name
        preds = net.run(
            [first_output_name], 
            {first_input_name: image}
        )[0]
    return preds


# ======================
# Main functions
# ======================
def recognize_from_image(net):
    # prepare input data
    image = Image.open(IMAGE_PATH).convert('L')
    image = pre_process(image)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds = predict(net, image)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        preds = predict(net, image)

    sim_pred = post_process(preds, len(preds), alphabet)
    print('============================================')
    print('String recognized from image is:', sim_pred)
    print('============================================')
    print('Script finished successfully.')


def recognize_from_video(net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    # if args.savepath != SAVE_IMAGE_PATH:
    #     f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    # else:
    #     writer = None

    while(True):
        ret, image = capture.read()
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        print('==============================================================')
        cv2.imshow('frame', image)
        image = Image.fromarray(np.uint8(image)).convert('L')
        image = pre_process(image)
        preds = predict(net, image)
        sim_pred = post_process(preds, len(preds), alphabet)
        print('String recognized from image is:', sim_pred)

        # save results
        # if writer is not None:
        #     writer.write(img)

    capture.release()
    cv2.destroyAllWindows()
    # if writer is not None:
    #     writer.release()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # model initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)


    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
