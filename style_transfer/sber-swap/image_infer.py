import numpy as np
import cv2

import mxnet as mx

M = np.array([
    [0.57142857, 0., 32.], [0., 0.57142857, 32.]
])

IM = np.array([
    [[1.75, -0., -56.], [-0., 1.75, -56.]]
])

onnx = False


def setup_mxnet():
    prefix = "./coordinate_reg/model/2d106det"
    ctx = mx.cpu()

    image_size = (192, 192)

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(
        for_training=False,
        data_shapes=[(
            'data', (1, 3, image_size[0], image_size[1])
        )])
    model.set_params(arg_params, aux_params)

    return model


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)

        new_pts[i] = new_pt[0:2]

    return new_pts


def get_landmarks(net_lmk, img):
    image_size = (192, 192)

    rimg = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    rimg = np.transpose(rimg, (2, 0, 1))  # 3*112*112, RGB

    data = np.expand_dims(rimg, axis=0)
    data = data.astype(np.float32)

    # feedforward
    # if not onnx:
    #     output = net_lmk.predict([data])
    # else:
    #     output = net_lmk.run(None, {'data': data})
    data = mx.nd.array(data)
    db = mx.io.DataBatch(data=(data,))
    net_lmk.forward(db, is_train=False)
    output = net_lmk.get_outputs()[-1].asnumpy()

    pred = output[0]

    pred = pred.reshape((-1, 2))
    pred[:, 0:2] += 1
    pred[:, 0:2] *= (image_size[0] // 2)

    pred = trans_points2d(pred, IM)

    return pred
