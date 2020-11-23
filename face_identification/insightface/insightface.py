import sys, os
import time
import argparse
import glob

import numpy as np
from numpy.linalg import norm
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, load_image  # noqa: E402C
from webcamera_utils import get_capture  # noqa: E402

from my_utils import PriorBox, decode, decode_landm, nms, face_align_norm_crop

# ======================
# Parameters
# ======================

WEIGHT_DET_PATH = './retinaface_resnet.onnx'
MODEL_DET_PATH = './retinaface_resnet.onnx.prototxt'
WEIGHT_REC_PATH = './arcface_r100_v1.onnx'
MODEL_REC_PATH = './arcface_r100_v1.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/insightface'

IMAGE_PATH = 'demo.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_SIZE = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='InsightFace model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH', default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '--det_thresh', type=float, default=0.02,
    help='det_thresh'
)
parser.add_argument(
    '--nms_thresh', type=float, default=0.4,
    help='nms_thresh'
)
parser.add_argument(
    '--top_k', type=int, default=5000,
    help='top_k'
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


def preprocess(img):
    img = np.float32(img)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def det_post_processing(im_height, im_width, loc, conf, landms):
    cfg_re50 = {
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
    }

    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    boxes = decode(loc[0], priors, cfg_re50['variance'])

    scale = np.array([im_width, im_height, im_width, im_height])
    boxes = boxes * scale
    scores = conf[0][:, 1]

    landms = decode_landm(landms[0], priors, cfg_re50['variance'])
    scale1 = np.array([
        im_width, im_height, im_width, im_height,
        im_width, im_height, im_width, im_height,
        im_width, im_height
    ])
    landms = landms * scale1

    inds = np.where(scores > args.det_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, args.nms_thresh)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms


def post_processing(bboxes, embs, ident_feats, img_width, img_height):
    detect_object = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        probs = bboxes[i, 4]
        emb = embs[i]
        metrics = ident_feats.dot(emb)
        cat = np.argmax(metrics)

        r = ailia.DetectorObject(
            category=cat,
            prob=probs,
            x=bbox[0] / img_width,
            y=bbox[1] / img_height,
            w=(bbox[2] - bbox[0]) / img_width,
            h=(bbox[3] - bbox[1]) / img_height,
        )
        detect_object.append(r)

    return detect_object


def load_identities(rec_model):
    names = []
    feats = []
    for path in glob.glob("identities/*.PNG"):
        name = ".".join(path.replace(os.sep, '/').split('/')[-1].split('.')[:-1])
        names.append(name)

        img = load_image(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        if not args.onnx:
            output = rec_model.predict({
                'data': img
            })[0]
        else:
            img_name = rec_model.get_inputs()[0].name
            fc1_name = rec_model.get_outputs()[0].name
            output = rec_model.run([fc1_name],
                                   {img_name: img})[0]

        embedding = output[0]
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm
        feats.append(normed_embedding)

    feats = np.vstack(feats)

    return names, feats


# ======================
# Main functions
# ======================


def predict(img, det_model, rec_model):
    # initial preprocesses
    im_height, im_width, _ = img.shape
    _img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = det_model.predict({
            'img': _img
        })
    else:
        img_name = det_model.get_inputs()[0].name
        loc_name = det_model.get_outputs()[0].name
        conf_name = det_model.get_outputs()[1].name
        landms_name = det_model.get_outputs()[2].name
        output = det_model.run([loc_name, conf_name, landms_name],
                               {img_name: _img})

    loc, conf, landms = output

    bboxes, landmarks = det_post_processing(im_height, im_width, loc, conf, landms)

    landms = []
    embs = []
    for i in range(bboxes.shape[0]):
        landmark = landmarks[i].reshape((5, 2))
        landms.append(landmark)

        _img = face_align_norm_crop(img, landmark=landmark)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _img = np.transpose(_img, (2, 0, 1))
        _img = np.expand_dims(_img, axis=0)
        _img = _img.astype(np.float32)
        if not args.onnx:
            output = rec_model.predict({
                'data': _img
            })[0]
        else:
            img_name = rec_model.get_inputs()[0].name
            fc1_name = rec_model.get_outputs()[0].name
            output = rec_model.run([fc1_name],
                                   {img_name: _img})[0]
        embedding = output[0]
        embedding_norm = norm(embedding)
        normed_embedding = embedding / embedding_norm
        embs.append(normed_embedding)

    embs = np.vstack(embs)

    return bboxes, embs


def recognize_from_image(filename, det_model, rec_model):
    # prepare input data
    img = load_image(filename)
    print(f'input image shape: {img.shape}')

    # load identities
    ident_names, ident_feats = load_identities(rec_model)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            bboxes, embs = predict(img, det_model, rec_model)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        bboxes, embs = predict(img, det_model, rec_model)

    img_height, img_width, _ = img.shape
    detect_object = post_processing(bboxes, embs, ident_feats, img_width, img_height)

    # plot result
    res_img = plot_results(detect_object, img, ident_names)

    # plot result
    cv2.imwrite(args.savepath, res_img)
    print('Script finished successfully.')


def recognize_from_video(video, det_model, rec_model):
    capture = get_capture(video)

    # load identities
    ident_names, ident_feats = load_identities(rec_model)

    while (True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        bboxes, embs = predict(frame, det_model, rec_model)
        img_height, img_width, _ = frame.shape
        detect_object = post_processing(bboxes, embs, ident_feats, img_width, img_height)

        # plot result
        res_img = plot_results(detect_object, frame, ident_names)

        # show
        cv2.imshow('frame', res_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    print("=== DET model ===")
    check_and_download_models(WEIGHT_DET_PATH, MODEL_DET_PATH, REMOTE_PATH)
    print("=== REC model ===")
    check_and_download_models(WEIGHT_REC_PATH, MODEL_REC_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    # initialize
    if not args.onnx:
        det_model = ailia.Net(MODEL_DET_PATH, WEIGHT_DET_PATH, env_id=env_id)
        rec_model = ailia.Net(MODEL_REC_PATH, WEIGHT_REC_PATH, env_id=env_id)
    else:
        import onnxruntime
        det_model = onnxruntime.InferenceSession(WEIGHT_DET_PATH)
        rec_model = onnxruntime.InferenceSession(WEIGHT_REC_PATH)

    if args.video is not None:
        recognize_from_video(args.video, det_model)
    else:
        recognize_from_image(args.input, det_model, rec_model)


if __name__ == '__main__':
    main()
