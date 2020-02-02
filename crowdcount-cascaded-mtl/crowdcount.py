import time
import os
import urllib.request

import cv2
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

import ailia


# Img size is decided when exporting model (modifiable when exporting)
width = 640
height = 480

# img_path = input("input image name: ")
img_path = "test.jpeg"
weight_path = "crowdcount.onnx"
model_path = "crowdcount.onnx.prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/crowd_count/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


def read_image(fname, shape=(width, height)):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32, copy=False)
    img = cv2.resize(img, shape)  
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))
    return img


# load image
img = read_image(img_path)
org_img = cv2.resize(
    # cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
    cv2.imread(img_path),
    (width, height)
)

# net initialize
env_id = ailia.get_gpu_environment_id()
print(env_id)
net = ailia.Net(model_path, weight_path, env_id=env_id)

# compute time
for i in range(3):
    start = int(round(time.time() * 1000))
    preds_ailia = net.predict(img)
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end - start))

# estimated crowd count
et_count = int(np.sum(preds_ailia))

# density map
density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
density_map = cv2.resize(density_map, (width, height))
heatmap = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
cv2.putText(
    heatmap,
    f'Est Count: {et_count}',
    (40, 440),  # position
    cv2.FONT_HERSHEY_SIMPLEX,  # font
    0.8,  # fontscale
    (255, 255, 255),  # color
    2  # thickness
)

res_img = np.hstack((org_img, heatmap))
cv2.imwrite('result.png', res_img)
print('Successfully finished !')


# # matplotlib & seaborn version to generate heatmap
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))
# plt.subplots_adjust(wspace=0.1)
# axes[0].imshow(org_img)
# axes[0].axis('off')

# sns.heatmap(density_map, cmap='RdBu_r', cbar=False)
# axes[1].text(
#     50, 600,
#     f'Est Count: {et_count}',
#     color='white',
#     fontsize='larger'
# )
# axes[1].axis('off')

# plt.savefig('result.png', bbox_inches='tight', pad_inches=0.1)
