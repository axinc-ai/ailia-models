import cv2
import sys
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/person-attributes-recognition-crossroad/"

IMAGE_PATH = 'input.jpg'
SLEEP_TIME = 0  # for video mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'person-attributes-recognition-crossroad model', IMAGE_PATH, None
)
parser.add_argument(
    '-m', '--model', default='0230', choices=['0230','0234']
)
args = update_parser(parser)

WEIGHT_PATH = "person-attributes-recognition-crossroad-{}.onnx".format(args.model)
MODEL_PATH = "person-attributes-recognition-crossroad-{}.onnx.prototxt".format(args.model)

MAX_CLASS_COUNT = 7
RECT_WIDTH = 640
RECT_HEIGHT = 20
RECT_MARGIN = 2
def plot_results(input_image, classifier, labels, top_k=MAX_CLASS_COUNT, logging=True):
    x = RECT_MARGIN
    y = RECT_MARGIN
    w = RECT_WIDTH
    h = RECT_HEIGHT

    if logging:
        print('==============================================================')
        print(f'class_count={top_k}')
    for idx in range(top_k):
        if logging:
            print(f'+ idx={idx}')
            print(f'  category={[idx]}['
                f'{labels[idx]} ]')
            print(f'  prob={classifier[idx]}')

        text = f'category=[{labels[idx]} ] prob={classifier[idx]}'

        color = hsv_to_rgb(256 * idx / (len(labels)+1), 128, 255)

        cv2.rectangle(input_image, (x, y), (x + w, y + h), color, thickness=-1)
        text_position = (x+4, y+int(RECT_HEIGHT/2)+4)

        color = (0,0,0)
        fontScale = 0.5

        cv2.putText(
            input_image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y + h + RECT_MARGIN


def print_results(classifier, labels, top_k=MAX_CLASS_COUNT):

    print('==============================================================')
    print(f'class_count={top_k}')
    for idx in range(top_k):
        #print(f'+ idx={idx}')
        print(f'  category={[idx]}['
              f'{labels[idx]} ]')
        print(f'  prob={classifier[idx]}')



# ======================
# Main functions
# ======================
def recognize_from_image(net):

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        raw_data = imread(image_path)
        input_data = raw_data.copy()
        if args.model == '0230':
            pass
        else:
            input_data = input_data.transpose(2,0,1)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                result = net.run(input_data)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            result = net.run(input_data)

        if args.model == '0230':
            classes = result[0][0][0][0]
        else:
            classes = result[0][0][:8]
        labels = ['is_male','has_bag','has_backpack','has_hat','has_longsleeves','has_longpants','has_longhair','has_coat_jacket']
        
        # show results
        print_results(classes,np.array(labels),7)

    logger.info('Script finished successfully.')


def recognize_from_video(net):

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # prepare input data
        input_data = frame
        if args.model == '0230':
            pass
        else:
            input_data = input_data.transpose(2,0,1)

        raw_w,raw_h = frame.shape[:2]

        # inference
        result = net.run(input_data)

        # get result
        if args.model == '0230':
            classes = result[0][0][0][0]
        else:
            classes = result[0][0][:8]
        labels = ['is_male','has_bag','has_backpack','has_hat','has_longsleeves','has_longpants','has_longhair','has_coat_jacket']
        
        plot_results(frame,classes,np.array(labels),7)

        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(None, WEIGHT_PATH, env_id=args.env_id)

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)


if __name__ == '__main__':
    main()
