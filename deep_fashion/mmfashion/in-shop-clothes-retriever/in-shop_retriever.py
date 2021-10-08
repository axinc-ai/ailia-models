import sys
import time
import os

import onnx
import onnxruntime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from pathlib import Path

import ailia

# import original modules
sys.path.append('../../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'in-shop_retriever.onnx'
MODEL_PATH = 'in-shop_retriever.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/mmfashion_retrieval/'

IMAGE_PATH = './06_1_front.jpg'
SAVE_IMAGE_PATH = 'output.png'

NORM_MEAN = [123.675, 116.28, 103.53]
NORM_STD = [58.395, 57.12, 57.375]

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('MMFashion model - In-shop Clothes Retriever', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-d', '--debug',
    action='store_true',
    help='Debugger'
)
parser.add_argument(
    '--onnx_runtime',
    action='store_true',
    help='Inference using onnx runtime'
)
parser.add_argument(
    '--gallery',
    type=str,
    help="Path to the root folder of the images' gallery"
)
parser.add_argument(
    '--img_file',
    type=str,
    help="Path to the images' filename of the gallery (.txt file)"
)
parser.add_argument(
    '--generate_img_file',
    action='store_true',
    help="Whether to generate the 'gallery_img.txt' file that contains the images' filename of the gallery."
)
parser.add_argument(
    '-k', '--topk',
    default=5, type=int,
    help='Retrieve the top k results'
)
args = update_parser(parser)
if not (args.gallery):
    parser.error('Argument --gallery is required.')

def check(onnx_model):
    onnx.checker.check_model(onnx_model)

def preprocess(img):
    h, w = img.shape[:2]

    # scale
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # normalize
    img = img.astype(np.float32)
    mean = np.array(NORM_MEAN)
    std = np.array(NORM_STD)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img

def recognize_from_image_onnx(filename, ort_session):
    # prepare input data
    img = load_image(filename)
    #logger.info((f'input image shape: {img.shape}')
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = preprocess(img)

    ort_inputs = {ort_session.get_inputs()[0].name: img}

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            ort_outs = ort_session.run(None, ort_inputs)
            pred = ort_outs[0]
            end = int(round(time.time() * 1000))
            logger.info(f'\tonnx runtime processing time {end - start} ms')
    else:
        ort_outs = ort_session.run(None, ort_inputs)
        pred = ort_outs[0]

    #logger.info(f'output shape: {pred.shape}')

def recognize_from_image(filename, net):
    # prepare input data
    img = load_image(filename)
    #logger.info(f'input image shape: {img.shape}')
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = preprocess(img)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            preds_ailia = net.predict(img)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
    else:
        preds_ailia = net.predict(img)

    #logger.info(f'output shape: {preds_ailia.shape}')
    search_gallery(net, preds_ailia, filename)

    savepath = get_savepath(args.savepath, filename)
    plt.savefig(savepath, bbox_inches='tight')
    logger.info(f'saved at : {savepath}')

def process_embeds(filenames, model):
    embeds = []
    logger.info('Exploring the gallery... (it may take a while)')

    for filename in filenames:
        # prepare input data
        img = load_image(os.path.join(args.gallery, filename)[:-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img)

        if len(embeds) % 500 == 0: 
            print(f"{len(embeds)}/{len(filenames)}")

        # inference
        embed = model.predict(img)
        embeds.append(embed)

        #if len(embeds) == 1000: break

    return embeds

def search_gallery(net, pred, filename=None):
    gallery_idx2im = {}
    gallery_imgs = open(args.img_file,'r').readlines()
    for i, img in enumerate(gallery_imgs):
        gallery_idx2im[i] = img.strip('\n')

    gallery_embeds = process_embeds(gallery_imgs, net)

    show_retrieved_images(pred, gallery_embeds, gallery_idx2im, args.topk, filename)

def show_topk_retrieved_images(retrieved_idxes, gallery_idx2im, input_image):
    fig = plt.figure(figsize=(15, 2.5))
    k=1
    n=len(retrieved_idxes)
    if input_image is not None:
        n+=1
        show_img(input_image, fig, 'Input image', n, k)

    for idx in retrieved_idxes:
        k+=1
        retrieved_img = gallery_idx2im[idx]
        filename = os.path.join(args.gallery, retrieved_img)
        print(filename)
        show_img(filename, fig, f'Top-{k-1} result', n, k)   

def show_retrieved_images(query_feat, gallery_embeds, gallery_idx2im, topk, filename):
    query_dist = []
    for i, feat in enumerate(gallery_embeds):
        cosine_dist = cosine(
            feat.reshape(1, -1), query_feat.reshape(1, -1))
        query_dist.append(cosine_dist)

    query_dist = np.array(query_dist)
    order = np.argsort(query_dist)

    logger.info('Retrieved Top%d Results' % topk)
    show_topk_retrieved_images(order[:topk], gallery_idx2im, filename)

def show_img(filename, fig, title, topk, idx):
    img = plt.imread(filename)
    ax = fig.add_subplot(1, topk, idx)
    plt.axis('off')
    imgplot = plt.imshow(img)
    ax.set_title(title)

"""
def recognize_from_video(filename, net):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds_ailia = net.predict(frame)
        search_gallery(net, preds_ailia)

        # save results
        if writer is not None:
            savepath = get_savepath(args.savepath, filename)
            plt.savefig(savepath, bbox_inches='tight')
            #writer.write(res_img)
        plt.show()

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')
"""

def generate_images_filename_txt(root):
    paths = list(Path(root).rglob("*.[jJ|pP][pP|nN][gG]"))
    filenames = [path.relative_to(root) for path in paths]
    folder = os.path.join(root, 'gallery_img.txt')
    with open(folder, 'w') as f:
        for filename in filenames:
            f.write("%s\n" % filename)
    logger.info(f'Gallery image filenames saved at : {folder}')
    args.img_file = folder

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    
    # debug
    if args.debug:
        onnx_model = onnx.load(WEIGHT_PATH)
        check(onnx_model)
    else:
        # generate gallery images filename .txt file
        if args.generate_img_file:
            generate_images_filename_txt(args.gallery)

        if not (args.img_file):
            parser.error('Either argument --img_file or --generate_img_file is required.')

        # net initialize
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

        if args.video is not None:
            # video mode
            #recognize_from_video(SAVE_IMAGE_PATH, net)
            pass
        else:
            # image mode
            if args.onnx_runtime:
                # onnx runtime
                ort_session = onnxruntime.InferenceSession(WEIGHT_PATH)
                # input image loop
                for image_path in args.input:
                    recognize_from_image_onnx(image_path, ort_session)
            else:
                # input image loop
                for image_path in args.input:
                    recognize_from_image(image_path, net)
    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()