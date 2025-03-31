# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Clustering pipelines"""


import random
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from pyannote_audio_utils.core import SlidingWindow, SlidingWindowFeature
from pyannote_audio_utils.pipeline import Pipeline
from pyannote_audio_utils.pipeline.parameter import Categorical, Integer, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist



# デバック
## ========================================================================
import csv

def write_array_to_csv(filename, data):
    """2次元配列をCSVファイルに保存する関数

    Args:
        data (list of list): 保存する2次元配列
        filename (str): 保存するファイル名
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
## ========================================================================



class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):
        super().__init__()
        self.metric = metric
        self.max_num_embeddings = max_num_embeddings
        self.constrained_assignment = constrained_assignment

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
    ):
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        return num_clusters, min_clusters, max_clusters

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter NaN embeddings and downsample embeddings

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.

        Returns
        -------
        filtered_embeddings : (num_embeddings, dimension) array
        chunk_idx : (num_embeddings, ) array
        speaker_idx : (num_embeddings, ) array
        """
        
        # print(embeddings.shape) #(21, 3, 256)

        # whether speaker is active
        active = np.sum(segmentations.data, axis=1) > 0
        # whether speaker embedding extraction went fine
        valid = ~np.any(np.isnan(embeddings), axis=2)

        # indices of embeddings that are both active and valid
        chunk_idx, speaker_idx = np.where(active * valid)

        # sample max_num_embeddings embeddings
        num_embeddings = len(chunk_idx)
        # print(self.max_num_embeddings) #inf
        # print(num_embeddings) #41
        if num_embeddings > self.max_num_embeddings:
            indices = list(range(num_embeddings))
            random.shuffle(indices)
            indices = sorted(indices[: self.max_num_embeddings])
            chunk_idx = chunk_idx[indices]
            speaker_idx = speaker_idx[indices]

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:
        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
    ):
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        centroids : (num_clusters, dimension)-shaped array
            Clusters centroids
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape
        # print(num_clusters) #3 OK
        # print(num_chunks) #21 OK
        # print(num_speakers) #3 OK
        # print(dimension) #256 OK
        

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]
        # print(train_embeddings.shape) #(41, 256)
        # write_array_to_csv("train_embeddings.csv", train_embeddings)

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )
        # print(centroids.shape) #(3, 256) OK
        # write_array_to_csv("centroids.csv", centroids) #順番は違うけどOK
        
        
        
        
        # print(embeddings.reshape([-1, dimension]).shape) #(63, 256)
        # print(cdist(embeddings.reshape([-1, dimension]), centroids, metric=self.metric).shape) #(63, 3)
        # write_array_to_csv("e2k_distance.csv" ,cdist(embeddings.reshape([-1, dimension]), centroids, metric=self.metric)) #OK

        e2k_distance = cdist(
            embeddings.reshape([-1, dimension]),
            centroids,
            metric=self.metric
        ).reshape([num_chunks, num_speakers, -1])
        # print(e2k_distance.shape) #(21, 3, 3)
        # print(e2k_distance[0]) #OK
        # print(e2k_distance[1]) #OK
        
        
        soft_clusters = 2 - e2k_distance
        # print(soft_clusters.shape) #(21, 3, 3)
        # print(soft_clusters[0]) #OK
        # print(soft_clusters[1]) #OK
        

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)
            
        # print(hard_clusters.shape) #(21, 3)
        # print(hard_clusters) #OK
        
        

        # NOTE: train_embeddings might be reassigned to a different cluster
        # in the process. based on experiments, this seems to lead to better
        # results than sticking to the original assignment.

        return hard_clusters, soft_clusters, centroids

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension) array
            Centroid vectors of each cluster
        """
        
        
        # print(embeddings.shape) #(21, 3, 256)
        # print(segmentations.data.shape) #(21, 589, 3)
        # write_array_to_csv("segmentations1.csv", segmentations.data[1]) #OK
        # write_array_to_csv("embeddings0.csv", embeddings[0]) OK

        train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )
        
        # print(train_embeddings.shape) #(41, 256)
        # print(len(train_chunk_idx)) #41
        # print(train_chunk_idx) #[ 0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20]
        # print(len(train_speaker_idx)) #41
        # print(train_speaker_idx) #[1 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2]
        # write_array_to_csv("train_embeddings.csv", train_embeddings) #OK

        num_embeddings, _ = train_embeddings.shape #(41, 256)
        
        # print(num_clusters) #2
        # print(min_clusters) #2
        # print(max_clusters) #2
        
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )
        
        # print(num_clusters) #2
        # print(min_clusters) #2
        # print(max_clusters) #2
        

        if max_clusters < 2:
            # do NOT apply clustering when min_clusters = max_clusters = 1
            # print("ax_clusters < 2") #呼ばれない
            
            # print(train_embeddings.shape) #(41, 256)
            
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            
            # print(hard_clusters.shape) #(21, 3)
            # print(soft_clusters.shape) #(21, 3, 1)
            # print(centroids.shape) #(1, 256)
            # print(centroids[0]) #-1.91367917e-01  9.46076556e-02  3.96066061e-02 -1.57084023e-01 ...
            
            return hard_clusters, soft_clusters, centroids

        # print(self.cluster) #<bound method AgglomerativeClustering.cluster of <pyannote_audio_utils.audio.pipelines.clustering.AgglomerativeClustering object at 0x307febee0>>
        
        # print(train_embeddings.shape) #(41, 256)
        # print(min_clusters) #1
        # print(max_clusters) #4
        # print(num_clusters) #None
        train_clusters = self.cluster(
            train_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
        )
        # print(train_clusters) #[0 2 0 0 2 0 2 0 2 0 2 2 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 0]


        # print(embeddings.shape) #(21, 3, 256)
        # print(self.constrained_assignment) #False
        # print(train_chunk_idx) #[ 0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20]
        # print(train_speaker_idx) #[1 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2]
        # print(train_clusters) #[0 2 0 0 2 0 2 0 2 0 2 2 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 0]
        hard_clusters, soft_clusters, centroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )
        # print(hard_clusters.shape) #(21, 3)
        # print(soft_clusters.shape) #(21, 3, 3)
        # print(centroids.shape) #(3, 256)
        # write_array_to_csv("hard_clusters.csv", hard_clusters)
        # write_array_to_csv("centroids.csv", centroids)
        # write_array_to_csv("soft_clusters[0].csv", soft_clusters[0])
        # write_array_to_csv("soft_clusters[1].csv", soft_clusters[1])
        # write_array_to_csv("soft_clusters[2].csv", soft_clusters[2])
        
        

        return hard_clusters, soft_clusters, centroids


class AgglomerativeClustering(BaseClustering):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold.
    min_cluster_size : int in range [1, 20]
        Minimum cluster size
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = False,
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        # print(self.threshold) #<pyannote_audio_utils.pipeline.parameter.Uniform object at 0x31652caf0>
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

        # minimum cluster size
        self.min_cluster_size = Integer(1, 20)
        

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: Optional[int] = None,
    ):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """
        
        
        # print(embeddings.shape) #(41, 256)
        # print(min_clusters) #1
        # print(num_clusters) #None
        # print(max_clusters) #41
        
        # 「-num 4」と指定した場合
        # print(min_clusters) #4
        # print(num_clusters) #4
        # print(max_clusters) #4
        

        num_embeddings, _ = embeddings.shape
        
        # print(self.min_cluster_size) #12

        # heuristic to reduce self.min_cluster_size when num_embeddings is very small
        # (0.1 value is kind of arbitrary, though)
        min_cluster_size = min(
            self.min_cluster_size, max(1, round(0.1 * num_embeddings))
        )
        
        # print(min_cluster_size) #4 OK
        

        # linkage function will complain when there is just one embedding to cluster
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # centroid, median, and Ward method only support "euclidean" metric
        # therefore we unit-normalize embeddings to somehow make them "euclidean"
        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            # print("if") #こっちが呼ばれる
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            # print(embeddings.shape) #(41, 256)
            # write_array_to_csv("norm_embeddings.csv", embeddings)
            
            # print(self.method) #centroid
            
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )
            
            # print(dendrogram.shape) #(40, 4)
            # write_array_to_csv("dendrogram.csv", dendrogram) #多分OK

        # other methods work just fine with any metric
        else:
            # print("else") #ここは呼ばれない
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )
            
        
        
        # apply the predefined threshold
        # print(dendrogram.shape) #(40, 4)
        # print(self.threshold) #0.7045654963945799
        clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        # print(clusters.shape) #(41,)
        # print(clusters + 1) #[1 3 1 1 5 1 4 1 3 1 3 3 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 1 2 2 1 2 1]
        # print(clusters) #[0 2 0 0 4 0 3 0 2 0 2 2 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 0]


        # split clusters into two categories based on their number of items:
        # large clusters vs. small clusters
        cluster_unique, cluster_counts = np.unique(
            clusters,
            return_counts=True,
        )
        # print(cluster_unique) #[0 1 2 3 4]
        # print(cluster_counts) #[21 14  4  1  1]
        
        # print(min_cluster_size) #4
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)
        
        # print(large_clusters) #[0 1 2]
        # print(num_large_clusters) #3

        # force num_clusters to min_clusters in case the actual number is too small
        if num_large_clusters < min_clusters:
            # print("num_large_clusters < min_clusters") #呼ばれない
            num_clusters = min_clusters

        # force num_clusters to max_clusters in case the actual number is too large
        elif num_large_clusters > max_clusters:
            # print("num_large_clusters > max_clusters") #呼ばれない
            num_clusters = max_clusters
            
        # print(num_clusters) #None
            

        # look for perfect candidate if necessary
        # ここの移植は後回し
        # print(num_clusters) #3
        # print(num_large_clusters) #3
        if num_clusters is not None and num_large_clusters != num_clusters:
            print("here") #呼ばれない
            # switch stopping criterion from "inter-cluster distance" stopping to "iteration index"
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)
            
            # print(_dendrogram.shape) #(40, 4)
            # write_array_to_csv("_dendrogram.csv", _dendrogram) # 見た目的には割とOK

            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1
            # print(best_iteration) #40

            # traverse the dendrogram by going further and further away
            # from the "optimal" threshold
            
            # print(self.threshold) #0.7045654963945799
            # print(Uniform(0.0, 2.0)) #<pyannote_audio_utils.pipeline.parameter.Uniform object at 0x32617ba30>
            debug_uniform = Uniform(0.0, 2.0)
            print(debug_uniform) #
            
            # print(dendrogram[:, 2]) #OK
            # [0.0749269 0.105544  0.111098  0.137631  0.185639  0.180859  0.186149
            # 0.191147  0.196887  0.197053  0.20312   0.203923  0.209653  0.228695
            # 0.230228  0.233008  0.236903  0.237292  0.270685  0.27205   0.270659
            # 0.287207  0.309903  0.331612  0.340587  0.369947  0.377406  0.408385
            # 0.410183  0.464871  0.467813  0.522171  0.525761  0.557698  0.600386
            # 0.659983  0.623513  0.752159  0.904207  0.922316 ]
            
            # print(np.abs(dendrogram[:, 2] - self.threshold))
            # [0.6296386 0.5990215 0.5934675 0.5669345 0.5189265 0.5237065 0.5184165
            # 0.5134185 0.5076785 0.5075125 0.5014455 0.5006425 0.4949125 0.4758705
            # 0.4743375 0.4715575 0.4676625 0.4672735 0.4338805 0.4325155 0.4339065
            # 0.4173585 0.3946625 0.3729535 0.3639785 0.3346185 0.3271595 0.2961805
            # 0.2943825 0.2396945 0.2367525 0.1823945 0.1788045 0.1468675 0.1041795
            # 0.0445825 0.0810525 0.0475935 0.1996415 0.2177505]
            
            # print(np.argsort(np.abs(dendrogram[:, 2] - self.threshold))) 
            #[35 37 36 34 33 32 31 38 39 30 29 28 27 26 25 24 23 22 21 19 18 20 17 16
            # 15 14 13 12 11 10  9  8  7  6  4  5  3  2  1  0]
            
            # print(_dendrogram[np.argsort(np.abs(dendrogram[:, 2] - self.threshold)), 3]) 
            #[ 2. 19.  5. 21.  3. 12.  3. 20. 41.  2.  9. 14.  2.  7.  9.  8.  6.  2.
            # 7.  2.  5.  8.  3.  6.  6.  4.  3.  2.  5.  5.  2.  3.  4.  2.  2.  4.
            # 2.  3.  2.  2.]
            # C++のdendrogramを入れた場合
            
            # print(min_cluster_size) #4

            for iteration in np.argsort(np.abs(dendrogram[:, 2] - self.threshold)):
                # only consider iterations that might have resulted
                # in changing the number of (large) clusters
                
                # print(self.threshold) #0.7045654963945799, 0.7045654963945799, 0.7045654963945799
                
                new_cluster_size = _dendrogram[iteration, 3]
                
                # print(iteration) #36, 37, 38
                # print(new_cluster_size) #5.0, 6.0, 20.0
                
                if new_cluster_size < min_cluster_size:
                    # print("continue") #呼ばれない
                    continue

                # estimate number of large clusters at considered iteration
                clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
                # print(clusters)
                # [0 2 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 0]
                # [0 3 0 0 1 0 1 0 1 0 1 1 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 0 2 2 0 2 0]
                # [0 5 0 0 1 0 2 0 3 0 3 3 0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 0 4 4 0 4 0]
                # [0 7 0 0 2 0 3 0 4 0 4 5 0 6 0 6 0 6 0 6 0 6 0 6 1 6 1 6 1 6 1 6 1 6 1 1 6 6 1 6 1]
                # [0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 0]
                # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
                # [ 0 10  2  1  5  3  6  3  7  3  7  8  3  9  3  9  3  9  3  9  3  9  3  9 4  9  4  9  4  9  4  9  4  9  4  4  9  9  4  9  4]
                # [ 0 11  2  1  6  3  7  3  8  3  8  9  3 10  3 10  3 10  3 10  3 10  3 10  4 10  4 10  4 10  4 10  4 10  4  4 10 10  4 10  5]
                # [ 0 13  2  1  6  3  7  3  8  3  9 10  3 11  3 11  3 11  3 11  3 11  3 11  4 11  4 12  4 12  4 12  4 12  4  4 12 12  4 12  5]
                
                
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                # print(cluster_unique)
                # [0 1 2]
                # [0 1 2 3]
                # [0 1 2 3 4 5]
                # [0 1 2 3 4 5 6 7]
                # [0 1]
                # [0]
                # [ 0  1  2  3  4  5  6  7  8  9 10]
                # [ 0  1  2  3  4  5  6  7  8  9 10 11]
                # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13]
                # print(cluster_counts)
                # [21 19  1]
                # [21  5 14  1]
                # [21  1  1  3 14  1]
                # [12  9  1  1  2  1 14  1]
                # [21 20]
                # [41]
                # [ 1  1  1  9  9  1  1  2  1 14  1]
                # [ 1  1  1  9  8  1  1  1  2  1 14  1]
                # [1 1 1 9 8 1 1 1 1 1 1 7 7 1]
                
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                # print(large_clusters)
                # [0 1]
                # [0 1 2]
                # [0 4]
                # [0 1 6]
                # [0 1]
                # [0]
                # [3 4 9]
                # [ 3  4 10]
                # [ 3  4 11 12]
                # [ 3  4 11 13]
                # [ 3  5 12 14]
                # [ 3  6 13 15]
                # [ 3  7 15 17]
                # [ 3  9 17 20]
                # [ 3  7 15 18]
                # [ 3  9 18 22]
                # [ 3 10 19 23]
                # [ 3 10 19 24]
                # [ 3 21 27]
                # [ 3 22 28]
                # [ 3 31]
                # [3]
                
                num_large_clusters = len(large_clusters)
                # print(num_large_clusters) #3, 3, 2, 3, 4 #OK

                # keep track of iteration that leads to the number of large clusters
                # as close as possible to the target number of clusters.
                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters

                # stop traversing the dendrogram as soon as we found a good candidate
                if num_large_clusters == num_clusters:
                    # print("break") #呼ばれる
                    break
                
            # print(best_num_large_clusters) #4
            # print(num_clusters) #4
            
            # print(min_cluster_size) #4

            # re-apply best iteration in case we did not find a perfect candidate
            if best_num_large_clusters != num_clusters:
                clusters = (
                    fcluster(_dendrogram, best_iteration, criterion="distance") - 1
                )
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)
                print(
                    f"Found only {num_large_clusters} clusters. Using a smaller value than {min_cluster_size} for `min_cluster_size` might help."
                )

        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters
        
        # print(min_cluster_size) #4

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters
        # print(small_clusters) #[ 0  1  2  5  6  7  8  9 10 13]
        

        # re-assign each small cluster to the most similar large cluster based on their respective centroids
        # print(large_clusters) #[ 3  4 11 12]
        # print(clusters) #[ 0 13  2  1  6  3  7  3  8  3  9 10  3 11  3 11  3 11  3 11  3 11  3 11 4 11  4 12  4 12  4 12  4 12  4  4 12 12  4 12  5]
        # print(embeddings.shape) #(41, 256)
        # write_array_to_csv("embeddings.csv", embeddings)
        large_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == large_k], axis=0)
                for large_k in large_clusters
            ]
        )
        # print(large_centroids.shape) #(4, 256)
        # write_array_to_csv("large_centroids.csv", large_centroids)
        
        small_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == small_k], axis=0)
                for small_k in small_clusters
            ]
        )
        # print(small_centroids.shape) #(10, 256)
        # write_array_to_csv("small_centroids.csv", small_centroids)
        
        # print(self.metric) #cosine
        # print(large_centroids.shape) #(4, 256)
        # print(small_centroids.shape) #(10, 256)
        centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        # print(centroids_cdist.shape) #(4, 10)
        # print(centroids_cdist)
        # [[0.19641003 0.16476353 0.2157328  0.25388748 0.39130301 0.73791351
        # 0.59812604 0.56090243 0.61215193 0.52166022]
        # [0.34149643 0.35807367 0.40102465 0.11402427 0.54068205 0.79429127
        # 0.62828912 0.5763684  0.62369865 0.58044486]
        # [0.59675619 0.60010363 0.7645834  0.67848269 0.57998222 0.49345192
        # 0.39426407 0.30638499 0.16898724 0.41270447]
        # [0.65106216 0.63983737 0.79086279 0.63123132 0.60710379 0.53245942
        # 0.46060271 0.38159056 0.26036441 0.50271472]]
        
        # print(small_clusters) #[ 0  1  2  5  6  7  8  9 10 13]
        # print(large_clusters) #[ 3  4 11 12]
        # print(clusters)
        # [ 0 13  2  1  6  3  7  3  8  3  9 10  3 11  3 11  3 11  3 11  3 11  3 11
        # 4 11  4 12  4 12  4 12  4 12  4  4 12 12  4 12  5]
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]
        # print(len(clusters)) #41
        # print(clusters) #[0 2 0 0 2 0 2 0 2 0 2 2 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 0]

        # re-number clusters from 0 to num_large_clusters
        _, clusters = np.unique(clusters, return_inverse=True)
        
        # print(len(clusters)) #41
        # print(clusters) #[0 2 0 0 0 0 2 0 2 0 2 2 0 2 0 2 0 2 0 2 0 2 0 2 1 2 1 3 1 3 1 3 1 3 1 1 3 3 1 3 1]
        
        return clusters


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering

