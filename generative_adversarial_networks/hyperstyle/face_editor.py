import numpy as np
import os

#from configs.paths_config import edit_paths
#from utils.common import tensor2im
from utils_ import np2im, INTERFACEGAN_EDITING

edit_paths = {
	'age': os.path.join(INTERFACEGAN_EDITING, 'age.npy'),
	'smile': os.path.join(INTERFACEGAN_EDITING, 'smile.npy'),
	'pose': os.path.join(INTERFACEGAN_EDITING, 'pose.npy'),
}

class FaceEditor:

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.interfacegan_directions = {
            'age': np.load(edit_paths['age']),
            'smile': np.load(edit_paths['smile']),
            'pose': np.load(edit_paths['pose'])
        }

    def apply_interfacegan(self, latents, weights_deltas, direction, factor=1, factor_range=None):
        edit_latents = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction
                edit_latents.append(edit_latent)
        else:
            edit_latents.append(latents + factor * direction)
        edit_latents = np.stack(edit_latents)
        return self._latents_to_image(edit_latents, weights_deltas)

    def _latents_to_image(self, all_latents, weights_deltas):
        sample_results, images = {}, []
        for _, sample_latents in enumerate(all_latents):
            sample_deltas = [d[0] if d is not None else None for d in weights_deltas]
            """
            images, _ = self.generator([sample_latents],
                                           weights_deltas=sample_deltas,
                                           randomize_noise=False,
                                           input_is_latent=True)
            """
            params = {str(i): sample_deltas[i-1] for i in range(2, 27)}
            params['[sample_latents]'] = sample_latents
            params['sample_deltas'] = sample_deltas[0]
            image, _ = self.generator.run(params)
            images.append(image)
        sample_results[0] = [np2im(image[0]) for image in images]
        return sample_results
