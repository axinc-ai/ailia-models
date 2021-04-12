from collections import defaultdict

import numpy as np
from PIL import Image as pimg

__all__ = ['ExtractInstances', 'RemapLabels', 'ColorizeLabels']


class ExtractInstances:
    def __init__(self, inst_map_to_id=None):
        self.inst_map_to_id = inst_map_to_id

    def __call__(self, example: dict):
        labels = np.int32(example['labels'])
        unique_ids = np.unique(labels)
        instances = defaultdict(list)
        for id in filter(lambda x: x > 1000, unique_ids):
            cls = self.inst_map_to_id.get(id // 1000, None)
            if cls is not None:
                instances[cls] += [labels == id]
        example['instances'] = instances
        return example


class RemapLabels:
    def __init__(self, mapping: dict, ignore_id, total=35):
        self.mapping = np.ones((max(total, max(mapping.keys())) + 1,), dtype=np.uint8) * ignore_id
        self.ignore_id = ignore_id
        for i in range(len(self.mapping)):
            self.mapping[i] = mapping[i] if i in mapping else ignore_id

    def _trans(self, labels):
        max_k = self.mapping.shape[0] - 1
        labels[labels > max_k] //= 1000
        labels = self.mapping[labels].astype(labels.dtype)
        return labels

    def __call__(self, example):
        if not isinstance(example, dict):
            return self._trans(example)
        if 'labels' not in example:
            return example
        ret_dict = {'labels': pimg.fromarray(self._trans(np.array(example['labels'])))}
        if 'original_labels' in example:
            ret_dict['original_labels'] = pimg.fromarray(self._trans(np.array(example['original_labels'])))
        return {**example, **ret_dict}


class ColorizeLabels:
    def __init__(self, color_info):
        self.color_info = np.array(color_info)

    def _trans(self, lab):
        R, G, B = [np.zeros_like(lab) for _ in range(3)]
        for l in np.unique(lab):
            mask = lab == l
            R[mask] = self.color_info[l][0]
            G[mask] = self.color_info[l][1]
            B[mask] = self.color_info[l][2]
        return np.stack((R, G, B), axis=-1).astype(np.uint8)

    def __call__(self, example):
        if not isinstance(example, dict):
            return self._trans(example)
        assert 'labels' in example
        return {**example, **{'labels': self._trans(example['labels']),
                              'original_labels': self._trans(example['original_labels'])}}
