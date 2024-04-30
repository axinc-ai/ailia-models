"""
@Author: Du Yunhao
@Filename: GSI.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/1 9:18
@Discription: Gaussian-smoothed interpolation
"""

import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


# 线性插值
def LinearInterpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # 按ID和帧排序
    output_ = input_.copy()
    '''线性插值'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # 同ID
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # 逐框插值
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:  # 不同ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_


# 高斯平滑
def GaussianSmooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)[:, 0]
        gpr.fit(t, y)
        yy = gpr.predict(t)[:, 0]
        gpr.fit(t, w)
        ww = gpr.predict(t)[:, 0]
        gpr.fit(t, h)
        hh = gpr.predict(t)[:, 0]
        output_.extend([
            [t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1, -1] for i in range(len(t))
        ])
    return np.array(output_)


# GSI
def GSInterpolation(track, interval, tau):
    li = LinearInterpolation(track, interval)
    gsi = GaussianSmooth(li, tau)
    return gsi
