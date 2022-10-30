import argparse

import numpy as np
import ailia
from PIL import Image

from torchvision.transforms import functional as F
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='ailia CosFace')
parser.add_argument('--target_image', type=str, default='image_target.jpg',
                    help='Test Target image path for prediction')
parser.add_argument('--id_image', type=str, default='image_id.jpg',
                    help='ID image path for prediction')
parser.add_argument('--onnx', type=str, default='cosface_sphere20.onnx',
                    help='Onnx model file path')
args = parser.parse_args()


def preprocessing(img_path):
    with open(img_path, 'rb') as f:
        original_img = Image.open(f).convert('RGB').resize((96, 112), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    img, img_flipped = transform(original_img), transform(F.hflip(original_img))
    img, img_flipped = img.unsqueeze(0).numpy(), img_flipped.unsqueeze(0).numpy()
    return {'original': img, 'flipped': img_flipped}


def extractDeepFeature(images, model):
    # features of original image and the flipped image are concatenated together
    # to compose the final face representation
    ft = np.concatenate((model.predict(images['original']), model.predict(images['flipped'])), 1)[0]
    return ft


def cosFace(query_img_path, id_img_path, model_path):
    # pre-processing
    img1 = preprocessing(query_img_path)
    img2 = preprocessing(id_img_path)

    # inference ( extract feature )
    model = ailia.Net(weight=model_path)
    f1 = extractDeepFeature(img1, model)
    f2 = extractDeepFeature(img2, model)

    # post-processing ( cosine distance )
    cosine_distance = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)
    return cosine_distance


if __name__ == '__main__':
    distance = cosFace(args.target_image,
                       args.id_image,
                       args.onnx)
    print(distance)
