import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402

# ======================
# Parameters
# ======================

WEIGHT_PATH_AIRPLANE = 'airplane.onnx'
MODEL_PATH_AIRPLANE = 'airplane.onnx.prototxt'
WEIGHT_PATH_BAG = 'bag.onnx'
MODEL_PATH_BAG = 'bag.onnx.prototxt'
WEIGHT_PATH_CAP = 'cap.onnx'
MODEL_PATH_CAP = 'cap.onnx.prototxt'
WEIGHT_PATH_CAR = 'car.onnx'
MODEL_PATH_CAR = 'car.onnx.prototxt'
WEIGHT_PATH_CHAIR = 'chair.onnx'
MODEL_PATH_CHAIR = 'chair.onnx.prototxt'
WEIGHT_PATH_EARPHONE = 'earphone.onnx'
MODEL_PATH_EARPHONE = 'earphone.onnx.prototxt'
WEIGHT_PATH_GUITAR = 'guitar.onnx'
MODEL_PATH_GUITAR = 'guitar.onnx.prototxt'
WEIGHT_PATH_KNIFE = 'knife.onnx'
MODEL_PATH_KNIFE = 'knife.onnx.prototxt'
WEIGHT_PATH_LAMP = 'lamp.onnx'
MODEL_PATH_LAMP = 'lamp.onnx.prototxt'
WEIGHT_PATH_LAPTOP = 'laptop.onnx'
MODEL_PATH_LAPTOP = 'laptop.onnx.prototxt'
WEIGHT_PATH_MOTORBIKE = 'motorbike.onnx'
MODEL_PATH_MOTORBIKE = 'motorbike.onnx.prototxt'
WEIGHT_PATH_MUG = 'mug.onnx'
MODEL_PATH_MUG = 'mug.onnx.prototxt'
WEIGHT_PATH_PISTOL = 'pistol.onnx'
MODEL_PATH_PISTOL = 'pistol.onnx.prototxt'
WEIGHT_PATH_ROCKET = 'rocket.onnx'
MODEL_PATH_ROCKET = 'rocket.onnx.prototxt'
WEIGHT_PATH_SKATEBOARD = 'skateboard.onnx'
MODEL_PATH_SKATEBOARD = 'skateboard.onnx.prototxt'
WEIGHT_PATH_TABLE = 'table.onnx'
MODEL_PATH_TABLE = 'table.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/pointnet/'

POINT_PATH_AIRPLANE = 'shapenetcore_dataset/Airplane/a1708ad923f3b51abbf3143b1cb6076a.pts'
POINT_PATH_BAG = 'shapenetcore_dataset/Bag/4e4fcfffec161ecaed13f430b2941481.pts'
POINT_PATH_CAP = 'shapenetcore_dataset/Cap/c7122c44495a5ac6aceb0fa31f18f016.pts'
POINT_PATH_CAR = 'shapenetcore_dataset/Car/1f1b5c7c01557c484354740e038a7994.pts'
POINT_PATH_CHAIR = 'shapenetcore_dataset/Chair/355fa0f35b61fdd7aa74a6b5ee13e775.pts'
POINT_PATH_EARPHONE = 'shapenetcore_dataset/Earphone/e33d6e8e39a75268957b6a4f3924d982.pts'
POINT_PATH_GUITAR = 'shapenetcore_dataset/Guitar/d546e034a6c659a425cd348738a8052a.pts'
POINT_PATH_KNIFE = 'shapenetcore_dataset/Knife/9d424831d05d363d870906b5178d97bd.pts'
POINT_PATH_LAMP = 'shapenetcore_dataset/Lamp/b8c87ad9d4930983a8d82fc8a3e54728.pts'
POINT_PATH_LAPTOP = 'shapenetcore_dataset/Laptop/4d3dde22f529195bc887d5d9a11f3155.pts'
POINT_PATH_MOTORBIKE = 'shapenetcore_dataset/Motorbike/9d3b07f4475d501e8249f134aca4c817.pts'
POINT_PATH_MUG = 'shapenetcore_dataset/Mug/10f6e09036350e92b3f21f1137c3c347.pts'
POINT_PATH_PISTOL = 'shapenetcore_dataset/Pistol/b1bbe535a833635d91f9af3df5b0c8fc.pts'
POINT_PATH_ROCKET = 'shapenetcore_dataset/Rocket/15474cf9caa757a528eba1f0b7744e9.pts'
POINT_PATH_SKATEBOARD = 'shapenetcore_dataset/Skateboard/f5d7698b5a57d61226e0640b67de606.pts'
POINT_PATH_TABLE = 'shapenetcore_dataset/Table/408c3db9b4ee6be2e9f3e9c758fef992.pts'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = argparse.ArgumentParser(
    description='Pixel-Link model'
)
parser.add_argument(
    '-i', '--input', metavar='POINT',
    default=None,
    help='The input point path.'
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
    '-c', '--choice_class', type=str, default='chair',
    choices=(
        'airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar',
        'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol',
        'rocket', 'skateboard', 'table'
    ),
    help='choice class'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = parser.parse_args()


# ======================
# Main functions
# ======================

def load_data(point_file, npoints=2500):
    point_set = np.loadtxt(point_file).astype(np.float32)
    choice = np.random.choice(len(point_set), npoints, replace=True)
    point_set = point_set[choice, :]

    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale

    return point_set


def predict(points, net):
    points = points.transpose(1, 0)
    points = np.expand_dims(points, axis=0)

    # feedforward
    if not args.onnx:
        net.set_input_shape(points.shape)
        output = net.predict({'point': points})
    else:
        point_name = net.get_inputs()[0].name
        pred_name = net.get_outputs()[0].name
        trans_name = net.get_outputs()[1].name
        output = net.run([pred_name, trans_name],
                         {point_name: points})

    pred, _ = output
    pred_choice = np.argmax(pred[0], axis=1)

    return pred_choice


def recognize_from_points(filename, net):
    # prepare input data
    points = load_data(filename)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            pred = predict(points, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        pred = predict(points, net)

    cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    pred_color = cmap[pred, :]

    showsz = 800
    points = points - points.mean(axis=0)
    radius = ((points ** 2).sum(axis=-1) ** 0.5).max()
    points /= (radius * 2.2) / showsz

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    ax.scatter(X, Y, Z, color=pred_color)

    # adjust plot scale
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() * 0.5
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig(args.savepath, dpi=120)
    plt.show()
    print('Script finished successfully.')


def main():
    rec_model = {
        'airplane': (WEIGHT_PATH_AIRPLANE, MODEL_PATH_AIRPLANE, POINT_PATH_AIRPLANE),
        'bag': (WEIGHT_PATH_BAG, MODEL_PATH_BAG, POINT_PATH_BAG),
        'cap': (WEIGHT_PATH_CAP, MODEL_PATH_CAP, POINT_PATH_CAP),
        'car': (WEIGHT_PATH_CAR, MODEL_PATH_CAR, POINT_PATH_CAR),
        'chair': (WEIGHT_PATH_CHAIR, MODEL_PATH_CHAIR, POINT_PATH_CHAIR),
        'earphone': (WEIGHT_PATH_EARPHONE, MODEL_PATH_EARPHONE, POINT_PATH_EARPHONE),
        'guitar': (WEIGHT_PATH_GUITAR, MODEL_PATH_GUITAR, POINT_PATH_GUITAR),
        'knife': (WEIGHT_PATH_KNIFE, MODEL_PATH_KNIFE, POINT_PATH_KNIFE),
        'lamp': (WEIGHT_PATH_LAMP, MODEL_PATH_LAMP, POINT_PATH_LAMP),
        'laptop': (WEIGHT_PATH_LAPTOP, MODEL_PATH_LAPTOP, POINT_PATH_LAPTOP),
        'motorbike': (WEIGHT_PATH_MOTORBIKE, MODEL_PATH_MOTORBIKE, POINT_PATH_MOTORBIKE),
        'mug': (WEIGHT_PATH_MUG, MODEL_PATH_MUG, POINT_PATH_MUG),
        'pistol': (WEIGHT_PATH_PISTOL, MODEL_PATH_PISTOL, POINT_PATH_PISTOL),
        'rocket': (WEIGHT_PATH_ROCKET, MODEL_PATH_ROCKET, POINT_PATH_ROCKET),
        'skateboard': (WEIGHT_PATH_SKATEBOARD, MODEL_PATH_SKATEBOARD, POINT_PATH_SKATEBOARD),
        'table': (WEIGHT_PATH_TABLE, MODEL_PATH_TABLE, POINT_PATH_TABLE),
    }
    WEIGHT_PATH, MODEL_PATH, POINT_PATH = rec_model[args.choice_class]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    recognize_from_points(args.input if args.input else POINT_PATH, net)


if __name__ == '__main__':
    main()
