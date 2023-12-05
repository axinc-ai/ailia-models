import numpy as np
import pickle

class ImageNet21kSemanticSoftmax:
    def __init__(self):
        self.class_list = np.load("class_list.npy",allow_pickle=True)
        with open("class_description.pkl","rb") as f:
            self.class_description = pickle.load(f)
        self.class_tree_list = np.load("class_tree.npy",allow_pickle=True)
        self.class_names = np.load("class_names.npy",allow_pickle=True)
        num_classes = len(self.class_tree_list)

        self.class_depth = np.zeros(num_classes)
        for i in range(num_classes):
            self.class_depth[i] = len(self.class_tree_list[i]) - 1
        max_depth = int(np.max(self.class_depth))

        # process semantic relations
        hist_tree = np.histogram(self.class_depth, bins=max_depth + 1, range=(0,max_depth))[0]
        hist_tree = hist_tree.astype(int)

        ind_list = []
        class_names_ind_list = []
        hirarchy_level_list = []

        cls = np.arange(num_classes)
        for i in range(max_depth):
            if hist_tree[i] > 1:
                hirarchy_level_list.append(i)
                ind_list.append(cls[self.class_depth == i])
                class_names_ind_list.append(self.class_names[ind_list[-1]])
        self.hierarchy_indices_list = ind_list
        self.hirarchy_level_list = hirarchy_level_list


    def split_logits_to_semantic_logits(self, logits):
        """
        split logits to 11 different hierarchies.

        :param self.self.hierarchy_indices_list: a list of size [num_of_hierarchies].
        Each element in the list is a tensor that contains the corresponding indices for the relevant hierarchy
        """
        semantic_logit_list = []
        for i, ind in enumerate(self.hierarchy_indices_list):
            logits_i = logits[:, ind]
            semantic_logit_list.append(logits_i)
        return semantic_logit_list
