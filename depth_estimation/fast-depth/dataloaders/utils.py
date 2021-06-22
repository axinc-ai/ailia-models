import numpy as np

import dataloaders.transforms as transforms

IHEIGHT, IWIDTH = 480, 640  # raw image size


def val_transform(rgb, depth_np, output_size):
    transform = transforms.Compose(
        [
            transforms.Resize((int(IWIDTH * (250.0 / IHEIGHT)), 250)),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(output_size),
        ]
    )
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype="float") / 255
    if depth_np is not None:
        depth_np = transform(depth_np)

    return rgb_np, depth_np
