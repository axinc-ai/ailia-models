import time
import os
import urllib.request

import numpy as np

import ailia


weight_path = "bert-base-uncased.onnx"
model_path = "bert-base-uncased.onnx.prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/bert_en/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


# load dataset
dummy_input = np.ones((1, 128))
dummy_inputs = np.array([dummy_input, dummy_input, dummy_input])
# dummy_inputs = (dummy_input, dummy_input, dummy_input)

# net initialize
env_id = ailia.get_gpu_environment_id()
print(env_id)
net = ailia.Net(model_path, weight_path, env_id=env_id)

# compute time
for i in range(1):
    start = int(round(time.time() * 1000))
    input_blobs = net.get_input_blob_list()
    for idx in input_blobs:
        net.set_input_blob_data(dummy_input, idx)
    net.update()
    preds_ailia = net.get_results()
    
    # preds_ailia = net.predict(dummy_input)[0]
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

print(f'[DEBUG] output shape: {preds_ailia[0].shape}')
print(f'[DEBUG] output shape: {preds_ailia[1].shape}')
print('Successfully finished!')
