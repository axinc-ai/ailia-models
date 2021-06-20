import math
import numpy as np


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return np.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.cpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.cpu_time = 0, 0

    def update(
        self,
        irmse,
        imae,
        mse,
        rmse,
        mae,
        absrel,
        lg10,
        delta1,
        delta2,
        delta3,
        cpu_time,
        data_time,
    ):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, cpu_time

    def evaluate(self, output, target):
        valid_mask = ((target > 0) + (output > 0)) > 0

        output = 1e3 * output[valid_mask]
        target = 1e3 * target[valid_mask]
        abs_diff = np.abs(output - target)

        self.mse = float((np.power(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float(np.abs(log10(output) - log10(target)).mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = np.maximum(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).astype(float).mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).astype(float).mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).astype(float).mean())
        self.data_time = 0
        self.cpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = np.abs(inv_output - inv_target)
        self.irmse = math.sqrt((np.power(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_cpu_time = 0, 0

    def update(self, result, cpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_cpu_time += n * cpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count,
            self.sum_imae / self.count,
            self.sum_mse / self.count,
            self.sum_rmse / self.count,
            self.sum_mae / self.count,
            self.sum_absrel / self.count,
            self.sum_lg10 / self.count,
            self.sum_delta1 / self.count,
            self.sum_delta2 / self.count,
            self.sum_delta3 / self.count,
            self.sum_cpu_time / self.count,
            self.sum_data_time / self.count,
        )
        return avg
