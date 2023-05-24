from typing import Dict, List, Union

import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class ClusterFeatures:
    """Basic handling of clustering features."""

    def __init__(
        self,
        features: ndarray,
        algorithm: str = 'kmeans',
        pca_k: int = None,
        random_state: int = 12345,
    ):
        """
        Cluster features constructor.

        :param features: the embedding matrix created by bert parent.
        :param algorithm: Which clustering algorithm to use.
        :param pca_k: If you want the features to be ran through pca, this is the components number.
        :param random_state: Random state.
        """
        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features

        self.algorithm = algorithm
        self.pca_k = pca_k
        self.random_state = random_state

    def _get_model(self, k: int) -> Union[GaussianMixture, KMeans]:
        """
        Retrieve clustering model.

        :param k: amount of clusters.
        :return: Clustering model.
        """
        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)

        return KMeans(n_clusters=k, random_state=self.random_state)

    def _get_centroids(self, model: Union[GaussianMixture, KMeans]) -> np.ndarray:
        """
        Retrieve centroids of model.

        :param model: Clustering model.
        :return: Centroids.
        """
        if self.algorithm == 'gmm':
            return model.means_

        return model.cluster_centers_

    def __find_closest_args(self, centroids: np.ndarray) -> Dict:
        """
        Find the closest arguments to centroid.

        :param centroids: Centroids to find closest.
        :return: Closest arguments.
        """
        centroid_min = 1e10
        cur_arg = -1
        args = {}
        used_idx = []

        for j, centroid in enumerate(centroids):

            for i, feature in enumerate(self.features):
                value = np.linalg.norm(feature - centroid)

                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min = value

            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e10
            cur_arg = -1

        return args

    def calculate_elbow(self, k_max: int) -> List[float]:
        """
        Calculates elbow up to the provided k_max.

        :param k_max: K_max to calculate elbow for.
        :return: The inertias up to k_max.
        """
        inertias = []

        for k in range(1, min(k_max, len(self.features))):
            model = self._get_model(k).fit(self.features)

            inertias.append(model.inertia_)

        return inertias

    def calculate_optimal_cluster(self, k_max: int) -> int:
        """
        Calculates the optimal cluster based on Elbow.

        :param k_max: The max k to search elbow for.
        :return: The optimal cluster size.
        """
        delta_1 = []
        delta_2 = []

        max_strength = 0
        k = 1

        inertias = self.calculate_elbow(k_max)

        for i in range(len(inertias)):
            delta_1.append(inertias[i] - inertias[i - 1] if i > 0 else 0.0)
            delta_2.append(delta_1[i] - delta_1[i - 1] if i > 1 else 0.0)

        for j in range(len(inertias)):
            strength = 0 if j <= 1 or j == len(inertias) - 1 else delta_2[j + 1] - delta_1[j + 1]

            if strength > max_strength:
                max_strength = strength
                k = j + 1

        return k

    def cluster(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        """
        Clusters sentences based on the ratio.

        :param ratio: Ratio to use for clustering.
        :param num_sentences: Number of sentences. Overrides ratio.
        :return: Sentences index that qualify for summary.
        """
        if num_sentences is not None:
            if num_sentences == 0:
                return []

            k = min(num_sentences, len(self.features))
        else:
            k = max(int(len(self.features) * ratio), 1)

        model = self._get_model(k).fit(self.features)

        centroids = self._get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)

        sorted_values = sorted(cluster_args.values())
        return sorted_values

    def __call__(self, ratio: float = 0.1, num_sentences: int = None) -> List[int]:
        """
        Clusters sentences based on the ratio.

        :param ratio: Ratio to use for clustering.
        :param num_sentences: Number of sentences. Overrides ratio.
        :return: Sentences index that qualify for summary.
        """
        return self.cluster(ratio)
