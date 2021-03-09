"""
 Deep white-balance editing main function (inference phase)
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
import deep_db_utils.utils as utls


def deep_wb(image, task='all', net_awb=None, net_t=None, net_s=None, device='cpu', s=656):
    # check image size
    image_resized = image.resize((round(image.width / max(image.size) * s), round(image.height / max(image.size) * s)))
    w, h = image_resized.size
    if w % 2 ** 4 == 0:
        new_size_w = w
    else:
        new_size_w = w + 2 ** 4 - w % 2 ** 4

    if h % 2 ** 4 == 0:
        new_size_h = h
    else:
        new_size_h = h + 2 ** 4 - h % 2 ** 4

    inSz = (new_size_w, new_size_h)
    if not ((w, h) == inSz):
        image_resized = image_resized.resize(inSz)

    image = np.array(image)
    image_resized = np.array(image_resized)
    img = image_resized.transpose((2, 0, 1))
    img = img / 255
    img = np.expand_dims(img, axis=0)

    print(img.shape)

    if task == 'all':
        output_awb = net_awb.run(img)[0]
        output_t = net_t.run(img)[0]
        output_s = net_s.run(img)[0]

        output_awb = output_awb[0]
        output_awb = output_awb.transpose((1, 2, 0))
        m_awb = utls.get_mapping_func(image_resized, output_awb)
        output_awb = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_awb))

        output_t = output_t[0]
        output_t = output_t.transpose((1, 2, 0))
        m_t = utls.get_mapping_func(image_resized, output_t)
        output_t = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_t))

        output_s = output_s[0]
        output_s = output_s.transpose((1, 2, 0))
        m_s = utls.get_mapping_func(image_resized, output_s)
        output_s = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_s))

        return output_awb, output_t, output_s