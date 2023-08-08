"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""

from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import numpy as np

INFINITY = 1e5


def fill_or_cut(x, former: bool):
    """
    :param x: input
    :param former: True代表该轨迹片段为连接时的前者
    """
    inputLen = 30

    lengthX, widthX = x.shape
    if lengthX >= inputLen:
        if former:
            x = x[-inputLen:]
        else:
            x = x[:inputLen]
    else:
        zeros = np.zeros((inputLen - lengthX, widthX))
        if former:
            x = np.concatenate((zeros, x), axis=0)
        else:
            x = np.concatenate((x, zeros), axis=0)
    return x


def transform(x1, x2):
    # fill or cut
    x1 = fill_or_cut(x1, True)
    x2 = fill_or_cut(x2, False)
    # min-max normalization
    min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
    max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
    subtractor = (max_ + min_) / 2
    divisor = (max_ - min_) / 2 + 1e-5
    x1 = (x1 - subtractor) / divisor
    x2 = (x2 - subtractor) / divisor
    # unsqueeze channel=1
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)

    return x1, x2


class AFLink:
    def __init__(self, model, thrT: tuple, thrS: int, thrP: float):
        self.thrP = thrP  # 预测阈值
        self.thrT = thrT  # 时域阈值
        self.thrS = thrS  # 空域阈值
        self.model = model  # 预测模型

    # 损失矩阵压缩
    def compression(self, cost_matrix, ids):
        # 行压缩
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # 列压缩
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # 矩阵压缩
        return matrix, ids_row, ids_col

    # 连接损失预测
    def predict(self, track1, track2):
        track1, track2 = transform(track1, track2)
        track1 = np.expand_dims(track1, axis=0)
        track2 = np.expand_dims(track2, axis=0)
        output = self.model.predict([track1, track2])
        score = output[0][0, 1]
        return 1 - score

    # 去重复: 即去除同一帧同一ID多个框的情况
    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)  # 保证帧号和ID号的唯一性
        return tracks[index]

    # 主函数
    def link(self, track):
        track = track[np.argsort(track[:, 0])]  # 按帧排序
        id2info = defaultdict(list)
        for row in track:
            f, i, x, y, w, h = row[:6]
            id2info[i].append([f, x, y, w, h])

        id2info = {k: np.array(v) for k, v in id2info.items()}

        num = len(id2info)  # 目标数量
        ids = np.array(list(id2info))  # 目标ID
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)  # L2距离
        cost_matrix = np.ones((num, num)) * INFINITY  # 损失矩阵

        '''计算损失矩阵'''
        for i, id_i in enumerate(ids):  # 前一轨迹
            for j, id_j in enumerate(ids):  # 后一轨迹
                if id_i == id_j:  # 禁止自娱自乐
                    continue

                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]
                if not self.thrT[0] <= fj - fi < self.thrT[1]:
                    continue
                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]):
                    continue
                cost = self.predict(info_i, info_j)
                if cost <= self.thrP:
                    cost_matrix[i, j] = cost

        '''二分图最优匹配'''
        id2id = dict()  # 存储临时匹配结果
        ID2ID = dict()  # 存储最终匹配结果
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k
        # print('  ', ID2ID.items())

        '''结果存储'''
        res = track.copy()
        for k, v in ID2ID.items():
            res[res[:, 1] == k, 1] = v
        res = self.deduplicate(res)

        return res
