import sys, os
import time
import argparse
import glob
from itertools import chain

import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C

# ======================
# Parameters
# ======================

WEIGHT_RESNET50_PATH = 'ft_ResNet50.onnx'
MODEL_RESNET50_PATH = 'ft_ResNet50.onnx.prototxt'
WEIGHT_FP16_PATH = 'fp16.onnx'
MODEL_FP16_PATH = 'fp16.onnx.prototxt'
WEIGHT_DENSE_PATH = 'ft_net_dense.onnx'
MODEL_DENSE_PATH = 'ft_net_dense.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/person_reid_baseline_pytorch/'

IMAGE_PATH = 'query/0342_c5s1_079123_00.jpg'
GALLERY_DIR = './gallery'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

FEATURE_RESULT_FILE = 'result.npy'
MARKET_1501_DROP_SAME_CAMERA_LABEL = True

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='UTS-Person-reID-Practical model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The query image path.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH', default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-g', '--gallery_dir', type=str, default=GALLERY_DIR,
    help='Gallery file directory'
)
parser.add_argument(
    '-d', '--data', type=str, default=None,
    help='Intermediate mat data file.'
)
parser.add_argument(
    '-m', '--model', type=str, default='resnet50',
    choices=('resnet50', 'fp16', 'dense'),
    help='the name of the model.'
)
parser.add_argument(
    '-b', '--batchsize', type=int, default=256,
    help='Batchsize.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = parser.parse_args()


# ======================
# Secondaty Functions
# ======================

class DataLoader(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        imgs = []
        file_list = [self.file_list[index]] if isinstance(index, int) else self.file_list[index]
        for filename in file_list:
            img = load_image(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = preprocess(img)
            img = img.transpose(2, 0, 1)
            imgs.append(img)

        imgs = np.stack(imgs)
        imgs = imgs[0] if isinstance(index, int) else imgs
        file_list = file_list[0] if isinstance(index, int) else file_list

        return imgs, file_list


def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255

    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    return img


def sort_img(query_feature, gallery_feature):
    query = query_feature.reshape(-1, 1)
    score = gallery_feature.dot(query)
    score = score.squeeze()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    return index


def get_id(img_path):
    filename = os.path.basename(img_path)
    label = filename[0:4]
    a = filename.split('c')
    camera_id = int(a[1][0])
    if label[:2] == '-1':
        label = -1
    else:
        label = int(label)

    return camera_id, label


def good_img(img_path, query_camera, query_label):
    camera, label = get_id(img_path)
    if label < 0:
        return False
    if camera == query_camera and label == query_label:
        return False
    return True


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# ======================
# Main functions
# ======================


def extract_feature(imgs, net):
    n, c, h, w = imgs.shape

    ff = np.zeros((n, 512))
    for i in range(2):
        if i == 1:
            # fliplr
            imgs = imgs[:, :, :, ::-1]

        if not args.onnx:
            net.set_input_shape(imgs.shape)
            outputs = net.predict({'img': imgs})[0]
        else:
            img_name = net.get_inputs()[0].name
            out_name = net.get_outputs()[0].name
            outputs = net.run([out_name],
                              {img_name: imgs})[0]
        ff += outputs

    fnorm = np.linalg.norm(ff, ord=2, axis=1, keepdims=True)
    fnorm = np.repeat(fnorm, ff.shape[1], axis=1)
    ff = ff / fnorm

    return ff


def recognize_from_image(query_path, net):
    image_list = [query_path]

    if args.data is None:
        ext_list = ["jpg", "png"]
        image_list.extend(
            chain.from_iterable([
                glob.glob(os.path.join(args.gallery_dir, "*." + ext)) for ext in ext_list
            ]))
        if len(image_list) == 1:
            print("GALLARY FILE (%s/*.jpg,*.png) not found." % args.gallery_dir)
            return

    start = int(round(time.time() * 1000))
    print('Start inference...')
    dataloader = DataLoader(image_list)
    features = []
    count = 0
    for i in range(0, len(dataloader), args.batchsize):
        imgs, _ = dataloader[i:i + args.batchsize]

        n, c, h, w = imgs.shape
        count += n
        print("%d/%d" % (count, len(dataloader)))

        outputs = extract_feature(imgs, net)
        features.append(outputs)

    features = np.vstack(features)

    end = int(round(time.time() * 1000))
    print(f'processing time {end - start} ms')

    query_feature = features[0]
    gallery_feature = features[1:]
    gallery_files = image_list[1:]

    if args.data:
        data = np.load(args.data, allow_pickle=True)
        data = data.item()
        gallery_feature = data['gallery_feature']
        gallery_files = data['gallery_file']
    else:
        data = {'gallery_feature': gallery_feature, 'gallery_file': gallery_files}
        np.save(FEATURE_RESULT_FILE, data)
        print("'%s' saved" % FEATURE_RESULT_FILE)

    index = sort_img(query_feature, gallery_feature)

    if MARKET_1501_DROP_SAME_CAMERA_LABEL:
        query_camera, query_label = get_id(query_path)
    else:
        query_label = None

    print('query_file:', query_path)
    print('Top 10 images are as follow:')
    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(query_path, 'query')

        count = 0
        for i in range(len(index)):
            img_path = gallery_files[index[i]]
            if MARKET_1501_DROP_SAME_CAMERA_LABEL and not good_img(img_path, query_camera, query_label):
                continue
            print(img_path)

            ax = plt.subplot(1, 11, count + 2)
            ax.axis('off')
            label = ""
            imshow(img_path)
            ax.set_title(
                '%d' % (count + 1),
                color='black' if not MARKET_1501_DROP_SAME_CAMERA_LABEL else \
                    'green' if label == query_label else 'red')

            count += 1
            if count >= 10:
                plt.show()
                break
    except RuntimeError:
        count = 0
        for i in range(10):
            img_path = gallery_files[index[i]]
            if MARKET_1501_DROP_SAME_CAMERA_LABEL and not good_img(img_path, query_camera, query_label):
                continue
            print(img_path)
            count += 1
            if count >= 10:
                break
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    fig.savefig(args.savepath)

    # plot result
    print('Script finished successfully.')


def main():
    dic_model = {
        'resnet50': (WEIGHT_RESNET50_PATH, MODEL_RESNET50_PATH),
        'fp16': (WEIGHT_FP16_PATH, MODEL_FP16_PATH),
        'dense': (WEIGHT_DENSE_PATH, MODEL_DENSE_PATH),
    }
    weight_path, model_path = dic_model[args.model]

    # model files check and download
    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    # initialize
    if not args.onnx:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        net = ailia.Net(model_path, weight_path, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(weight_path)

    recognize_from_image(args.input, net)


if __name__ == '__main__':
    main()
